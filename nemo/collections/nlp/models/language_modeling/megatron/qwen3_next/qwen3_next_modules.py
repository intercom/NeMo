# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import divide
from torch import Tensor


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along specified dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunk-based gated delta rule implementation."""
    batch_size, num_heads, seq_len, head_dim = query.shape

    # L2 normalize query and key if specified
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Convert to float32 and transpose for processing
    query = query.transpose(1, 2).contiguous().to(torch.float32)  # [B, S, H, D]
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    num_chunks = math.ceil(seq_len / chunk_size)
    output = torch.zeros_like(value)

    # Initialize state
    if initial_state is not None:
        state = initial_state.clone()
    else:
        state = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=query.device)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, seq_len)

        # Extract chunk
        q_chunk = query[:, start_idx:end_idx]  # [B, C, H, D]
        k_chunk = key[:, start_idx:end_idx]
        v_chunk = value[:, start_idx:end_idx]
        g_chunk = g[:, start_idx:end_idx]
        beta_chunk = beta[:, start_idx:end_idx]

        chunk_len = end_idx - start_idx

        # Compute decay matrix for this chunk
        decay = torch.exp(-beta_chunk.unsqueeze(-1))  # [B, C, H, 1]

        # Create causal mask for chunk
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, device=query.device))

        # Process chunk with gated delta rule
        chunk_output = torch.zeros_like(v_chunk)

        for t in range(chunk_len):
            # Apply gating
            gated_k = k_chunk[:, t] * g_chunk[:, t].unsqueeze(-1)  # [B, H, D]

            # Update state
            state = decay[:, t].unsqueeze(-1) * state + torch.einsum('bhd,bhe->bhde', gated_k, v_chunk[:, t])

            # Compute output
            chunk_output[:, t] = torch.einsum('bhd,bhde->bhe', q_chunk[:, t], state)

        output[:, start_idx:end_idx] = chunk_output

    # Transpose back to original format
    output = output.transpose(1, 2).contiguous()  # [B, H, S, D]

    final_state = state if output_final_state else None
    return output, final_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Recurrent gated delta rule implementation."""
    batch_size, num_heads, seq_len, head_dim = query.shape

    # L2 normalize query and key if specified
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Convert to float32
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)

    output = torch.zeros_like(value)

    # Initialize state
    if initial_state is not None:
        state = initial_state.clone()
    else:
        state = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=torch.float32, device=query.device)

    # Process sequence recurrently
    for t in range(seq_len):
        # Compute decay
        decay = torch.exp(-beta[:, :, t].unsqueeze(-1))  # [B, H, 1]

        # Apply gating to key
        gated_k = key[:, :, t] * g[:, :, t].unsqueeze(-1)  # [B, H, D]

        # Update state with gated delta rule
        state = decay * state + torch.einsum('bhd,bhe->bhde', gated_k, value[:, :, t])

        # Compute output
        output[:, :, t] = torch.einsum('bhd,bhde->bhe', query[:, :, t], state)

    final_state = state if output_final_state else None
    return output, final_state


class Qwen3NextLinearAttention(MegatronModule):
    """
    Qwen3-Next Linear Attention with Gated Delta Rule.
    This implements the linear attention mechanism from Qwen3-Next architecture.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # Model parallel setup
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # Linear attention specific parameters
        self.chunk_size = getattr(config, 'linear_attention_chunk_size', 64)
        self.use_recurrent = getattr(config, 'linear_attention_use_recurrent', False)
        self.clip_val = getattr(config, 'linear_attention_clip_val', 5.0)

        # Projection layers - following NeMo patterns
        self.query = ColumnParallelLinear(
            config.hidden_size,
            projection_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        self.key = ColumnParallelLinear(
            config.hidden_size,
            projection_size // (config.num_attention_heads // config.num_query_groups),
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        self.value = ColumnParallelLinear(
            config.hidden_size,
            projection_size // (config.num_attention_heads // config.num_query_groups),
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        # Gating and beta parameters
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            projection_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        self.beta_proj = ColumnParallelLinear(
            config.hidden_size,
            projection_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        # Output projection
        self.dense = RowParallelLinear(
            projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
        )

        # Normalization layers
        self.q_norm = TENorm(config, self.hidden_size_per_attention_head)
        self.k_norm = TENorm(config, self.hidden_size_per_attention_head)

        # Dropout
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass for linear attention."""

        # Get input dimensions
        seq_length, batch_size, hidden_size = hidden_states.size()

        # Project to QKV
        query, _ = self.query(hidden_states)
        key, _ = self.key(hidden_states)
        value, _ = self.value(hidden_states)

        # Project gating parameters
        g, _ = self.g_proj(hidden_states)
        beta, _ = self.beta_proj(hidden_states)

        # Reshape for multi-head attention
        new_tensor_shape = (seq_length, batch_size, self.num_attention_heads_per_partition, -1)
        query = query.view(*new_tensor_shape).transpose(0, 1)  # [B, S, H, D]

        # Handle grouped query attention
        kv_heads = self.num_query_groups_per_partition
        kv_tensor_shape = (seq_length, batch_size, kv_heads, -1)
        key = key.view(*kv_tensor_shape).transpose(0, 1)
        value = value.view(*kv_tensor_shape).transpose(0, 1)

        # Expand key/value for GQA
        if self.num_attention_heads_per_partition // kv_heads > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // kv_heads, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // kv_heads, dim=2
            )

        # Reshape g and beta
        g = g.view(*new_tensor_shape).transpose(0, 1)
        beta = beta.view(*new_tensor_shape).transpose(0, 1)

        # Apply normalization to query and key
        batch_size, seq_len, num_heads, head_dim = query.shape
        query = query.contiguous().view(-1, head_dim)
        key = key.contiguous().view(-1, head_dim)

        query = self.q_norm(query).view(batch_size, seq_len, num_heads, head_dim)
        key = self.k_norm(key).view(batch_size, seq_len, num_heads, head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            query = rotary_pos_emb(query)
            key = rotary_pos_emb(key)

        # Transpose for processing: [B, S, H, D] -> [B, H, S, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # Clip beta values
        beta = torch.clamp(beta, min=0, max=self.clip_val)

        # Apply gated delta rule
        if self.use_recurrent or seq_len <= self.chunk_size:
            context_layer, _ = torch_recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            context_layer, _ = torch_chunk_gated_delta_rule(
                query, key, value, g, beta,
                chunk_size=self.chunk_size,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        # Apply dropout
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.attention_dropout(context_layer)
        else:
            context_layer = self.attention_dropout(context_layer)

        # Reshape and transpose back: [B, H, S, D] -> [S, B, H*D]
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_shape = (seq_length, batch_size, self.hidden_size_per_partition)
        context_layer = context_layer.view(*new_context_shape)

        # Final output projection
        output, _ = self.dense(context_layer)

        return output