# Qwen3-Next Integration for NeMo

This directory contains the implementation of Qwen3-Next architecture support for NeMo's Megatron framework.

## Overview

Qwen3-Next is an advanced transformer architecture that supports hybrid attention mechanisms, combining both traditional multi-head attention and linear attention layers within the same model. This allows for better scaling to longer sequences while maintaining the expressivity needed for complex reasoning tasks.

## Key Features

- **Hybrid Attention Architecture**: Mix of full attention and linear attention layers
- **Gated Delta Rule**: Efficient linear attention mechanism with gating
- **Chunk-based Processing**: Memory-efficient processing for long sequences
- **Enhanced RoPE Support**: Advanced rotary position embeddings
- **NeMo Integration**: Full integration with NeMo's Megatron framework

## Files

- `qwen3_next_modules.py`: Core modules including linear attention implementation
- `qwen3_next_spec.py`: Layer specifications for both full and linear attention
- `qwen3_next_layer.py`: Custom transformer layer supporting hybrid attention
- `__init__.py`: Package initialization

## Usage

### Basic Usage

To use Qwen3-Next in your NeMo model configuration, set the `transformer_layer_spec` to `"megatron_qwen3_next"`:

```python
model_config = {
    "transformer_layer_spec": "megatron_qwen3_next",
    "layer_types": ["full", "linear", "full", "linear"],  # Hybrid attention pattern
    "linear_attention_clip_val": 5.0,
    "linear_attention_chunk_size": 64,
    "attention_head_type": "MQA",  # or "GQA"
}
```

### Layer Types Configuration

The `layer_types` parameter controls which attention mechanism each layer uses:
- `"full"`: Standard multi-head attention
- `"linear"`: Linear attention with gated delta rule

Example patterns:
```python
# All linear attention
layer_types = ["linear"] * num_layers

# Alternating pattern
layer_types = ["full", "linear"] * (num_layers // 2)

# Custom pattern for different model sizes
layer_types = ["full", "full", "linear", "linear", "full", "linear"]
```

### Configuration Parameters

- `layer_types`: List of layer types (required for hybrid models)
- `linear_attention_clip_val`: Clipping value for beta parameters (default: 5.0)
- `linear_attention_chunk_size`: Chunk size for chunk-based processing (default: 64)
- `linear_attention_use_recurrent`: Force recurrent processing (default: False)
- `attention_head_type`: "MQA" or "GQA" for query group attention

## Architecture Details

### Linear Attention

The linear attention mechanism uses a gated delta rule that processes sequences more efficiently than traditional attention:

1. **Gated Projections**: Additional g and beta projections for gating and decay
2. **L2 Normalization**: Query and key normalization for stability
3. **Delta Rule Updates**: Efficient state updates using exponential decay
4. **Chunk Processing**: Memory-efficient processing for long sequences

### Memory Complexity

- Traditional Attention: O(n²) memory complexity
- Linear Attention: O(n) memory complexity
- Hybrid Model: Balanced trade-off between expressivity and efficiency

## Integration with NeMo

The Qwen3-Next architecture is fully integrated with NeMo's existing infrastructure:

- Model parallel support via tensor parallelism
- Mixed precision training support
- Sequence parallelism compatibility
- Standard NeMo configuration system
- HuggingFace model conversion support (planned)

## Performance Considerations

- Linear attention layers are more efficient for long sequences (>1K tokens)
- Full attention layers provide better performance for complex reasoning
- Hybrid models balance efficiency and capability
- Chunk size can be tuned based on available memory

## Future Enhancements

- [ ] Advanced caching for inference
- [ ] HuggingFace model conversion utilities
- [ ] MoE (Mixture of Experts) integration
- [ ] Flash Attention optimization
- [ ] Custom CUDA kernels for linear attention

## References

- [Qwen3-Next Paper](https://arxiv.org/abs/XXXX.XXXXX) (placeholder)
- [Transformers Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_next)
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/)
- [Intercom NeMo Fork](https://github.com/intercom/NeMo)