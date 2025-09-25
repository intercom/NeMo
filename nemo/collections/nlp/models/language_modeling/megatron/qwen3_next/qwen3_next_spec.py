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

from typing import List, Optional

from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer import ModuleSpec, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from .qwen3_next_layer import Qwen3NextTransformerLayer, build_qwen3_next_layer_with_layer_type
from .qwen3_next_modules import Qwen3NextLinearAttention


def get_qwen3_next_layer_spec(layer_type: str = "full") -> ModuleSpec:
    """
    Get the layer specification for Qwen3-Next architecture.

    Args:
        layer_type: "full" for standard multi-head attention, "linear" for linear attention

    Returns:
        ModuleSpec for the specified layer type
    """
    if layer_type == "linear":
        return _get_qwen3_next_linear_attention_layer_spec()
    else:
        return _get_qwen3_next_full_attention_layer_spec()


def get_qwen3_next_layer_specs(layer_types: List[str]) -> List[ModuleSpec]:
    """
    Get layer specifications for a list of layer types.
    This supports the hybrid attention architecture of Qwen3-Next.

    Args:
        layer_types: List of layer types, e.g., ["full", "linear", "full", "linear"]

    Returns:
        List of ModuleSpec objects for each layer type
    """
    return [get_qwen3_next_layer_spec(layer_type) for layer_type in layer_types]


def _get_qwen3_next_full_attention_layer_spec() -> ModuleSpec:
    """Get layer spec for standard multi-head attention layers."""
    return ModuleSpec(
        module=Qwen3NextTransformerLayer,
        params={"layer_type": "full"},
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    # Use standard dot product attention for full attention layers
                    core_attention=None,  # Will use default TEDotProductAttention
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def _get_qwen3_next_linear_attention_layer_spec() -> ModuleSpec:
    """Get layer spec for linear attention layers."""
    return ModuleSpec(
        module=Qwen3NextTransformerLayer,
        params={"layer_type": "linear"},
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=Qwen3NextLinearAttention,
                params={"attn_mask_type": AttnMaskType.causal, "attention_type": "self"},
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def create_qwen3_next_transformer_layer_spec(
    layer_number: int,
    layer_types: List[str],
    config=None
) -> ModuleSpec:
    """
    Create a transformer layer spec for a specific layer number based on the layer_types configuration.

    Args:
        layer_number: The layer index (0-based)
        layer_types: List of layer types for all layers
        config: Optional transformer config

    Returns:
        ModuleSpec for the specified layer
    """
    if layer_number >= len(layer_types):
        # Default to full attention if layer_number exceeds layer_types list
        layer_type = "full"
    else:
        layer_type = layer_types[layer_number]

    return get_qwen3_next_layer_spec(layer_type)


# Main function to get the appropriate layer specification
def get_qwen3_next_layer_spec_with_config(config, layer_number: int = 0) -> ModuleSpec:
    """
    Get layer specification based on model configuration.

    Args:
        config: Model configuration containing layer_types
        layer_number: Layer index for hybrid attention models

    Returns:
        ModuleSpec for the specified layer
    """
    # Check if this is a hybrid model with different layer types
    if hasattr(config, 'layer_types') and config.layer_types is not None:
        layer_types = config.layer_types
        if layer_number < len(layer_types):
            layer_type = layer_types[layer_number]
        else:
            # Default to full attention if not specified
            layer_type = "full"
    else:
        # Default to full attention for standard models
        layer_type = "full"

    return get_qwen3_next_layer_spec(layer_type)