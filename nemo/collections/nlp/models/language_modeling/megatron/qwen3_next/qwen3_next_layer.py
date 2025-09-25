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

from typing import Optional, Union

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from torch import Tensor


class Qwen3NextTransformerLayer(TransformerLayer):
    """
    A Qwen3-Next Transformer Layer that can dynamically use either standard attention
    or linear attention based on the layer_type configuration.

    This extends the standard TransformerLayer to support the hybrid attention
    mechanism required by Qwen3-Next architecture.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        layer_type: str = "full",
        hidden_dropout: float = None,
    ):
        self.layer_type = layer_type
        super().__init__(config, submodules, layer_number, hidden_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        inference_params=None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Forward pass supporting both full and linear attention modes."""

        # Store input for residual connection
        residual = hidden_states

        # Pre-attention LayerNorm
        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attention_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=context,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # Attention residual connection with bias-dropout-add
        if self.self_attn_bda is not None:
            attention_output_with_bias = self.self_attn_bda(attention_output, None)
            if attention_output_with_bias.shape != residual.shape:
                # Handle potential shape mismatch
                attention_output_with_bias = attention_output_with_bias.view(residual.shape)

            hidden_states = residual + attention_output_with_bias
        else:
            hidden_states = residual + attention_output

        # Pre-MLP LayerNorm
        residual = hidden_states
        if self.pre_mlp_layernorm is not None:
            hidden_states = self.pre_mlp_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states, inference_params=inference_params)

        # MLP residual connection with bias-dropout-add
        if self.mlp_bda is not None:
            mlp_output_with_bias = self.mlp_bda(mlp_output, None)
            if mlp_output_with_bias.shape != residual.shape:
                # Handle potential shape mismatch
                mlp_output_with_bias = mlp_output_with_bias.view(residual.shape)

            hidden_states = residual + mlp_output_with_bias
        else:
            hidden_states = residual + mlp_output

        # Post-attention LayerNorm for some architectures (like old Falcon)
        if hasattr(self, 'post_self_attn_layernorm') and self.post_self_attn_layernorm is not None:
            hidden_states = self.post_self_attn_layernorm(hidden_states)

        return hidden_states


def build_qwen3_next_layer_with_layer_type(
    config: TransformerConfig,
    submodules: TransformerLayerSubmodules,
    layer_number: int,
    layer_type: str = "full"
) -> Qwen3NextTransformerLayer:
    """
    Factory function to build a Qwen3-Next transformer layer with specified type.

    Args:
        config: TransformerConfig containing model configuration
        submodules: TransformerLayerSubmodules defining the layer components
        layer_number: Layer index in the transformer stack
        layer_type: "full" for standard attention, "linear" for linear attention

    Returns:
        Configured Qwen3NextTransformerLayer
    """
    return Qwen3NextTransformerLayer(
        config=config,
        submodules=submodules,
        layer_number=layer_number,
        layer_type=layer_type,
        hidden_dropout=config.hidden_dropout,
    )