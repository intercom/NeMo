# Qwen3-Next Training Guide

This guide shows how to train a 3B parameter Qwen3-Next model with 1B active parameters using FineWeb dataset.

## Prerequisites

1. **NeMo Installation**: Ensure NeMo is installed with all dependencies
2. **GPU Setup**: 8x GPUs recommended (can be adjusted in config)
3. **Storage**: ~100GB for preprocessed FineWeb sample-10BT subset

## Step 1: Preprocess FineWeb Dataset

Convert FineWeb from HuggingFace to NeMo's binary format:

```bash
python scripts/nlp_language_modeling/preprocess_fineweb_for_qwen3_next.py \
    --output-prefix ./data/fineweb_qwen3 \
    --subset sample-10BT \
    --tokenizer-model Qwen/Qwen2-1.5B \
    --max-docs 1000000 \
    --workers 8 \
    --append-eod
```

This will create:
- `./data/fineweb_qwen3_text_document.bin` (binary data)
- `./data/fineweb_qwen3_text_document.idx` (index file)

### Preprocessing Options

- `--subset sample-10BT`: Use 10BT sample (faster) or `sample-350BT` (full dataset)
- `--max-docs 1000000`: Limit documents for experimentation
- `--workers 8`: Parallel processing (adjust based on CPU cores)

## Step 2: Start Training

Launch training with the Qwen3-Next configuration:

```bash
python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=. \
    --config-name=qwen3_next_3b_fineweb_training.yaml
```

### Key Configuration Features

**Hybrid Attention Architecture:**
- 24 layers total: 6 full attention + 18 linear attention
- Linear attention provides O(n) complexity for long sequences
- Full attention maintains reasoning capability

**Model Parameters:**
- 3B total parameters, ~1B active during inference
- 2048 hidden size, 16 attention heads
- 4096 sequence length with RoPE position embeddings

**Training Setup:**
- Tensor parallelism across 2 GPUs
- Micro batch size: 2, Global batch size: 64
- BFloat16 mixed precision
- Gradient accumulation: 4 steps

## Step 3: Monitor Training

**Logs Location:** `./logs/qwen3_next_3b_fineweb/`

**Key Metrics to Watch:**
- `train_loss`: Should decrease steadily
- `val_loss`: Validation loss for overfitting detection
- `lr`: Learning rate schedule
- `throughput_per_gpu`: Training speed

**Checkpoints:** Saved in experiment directory with best validation loss

## Architecture Details

### Qwen3-Next Hybrid Attention

The model uses two attention mechanisms:

1. **Full Attention Layers** (6 layers):
   - Standard multi-head attention
   - O(n²) complexity but full expressivity
   - Used for complex reasoning

2. **Linear Attention Layers** (18 layers):
   - Gated delta rule mechanism
   - O(n) complexity for efficiency
   - Chunk-based processing for long sequences

### Layer Configuration

```yaml
layer_types: [
  "full", "linear", "linear", "full",      # Layers 0-3
  "linear", "linear", "full", "linear",    # Layers 4-7
  "linear", "linear", "linear", "full",    # Layers 8-11
  "linear", "linear", "full", "linear",    # Layers 12-15
  "linear", "linear", "linear", "full",    # Layers 16-19
  "linear", "linear", "full", "linear"     # Layers 20-23
]
```

## Memory and Performance

**Expected Memory Usage:**
- ~24GB per GPU for model parameters
- ~8GB additional for gradients and optimizer states
- Total: ~32GB per GPU (fits on A100 40GB)

**Training Speed:**
- ~2000 tokens/sec/GPU with sequence length 4096
- ~50 hours for 50K steps on 8x A100

**Scaling Options:**
- Reduce `micro_batch_size` if OOM
- Increase `tensor_model_parallel_size` for larger GPUs
- Adjust `accumulate_grad_batches` for different batch sizes

## Customization

### Different Dataset Sizes

**Small Scale Testing:**
```bash
# Process only 10K documents
python scripts/nlp_language_modeling/preprocess_fineweb_for_qwen3_next.py \
    --max-docs 10000 \
    --subset sample-10BT
```

**Full Scale Training:**
```bash
# Use full sample-350BT subset
python scripts/nlp_language_modeling/preprocess_fineweb_for_qwen3_next.py \
    --subset sample-350BT \
    --max-docs 10000000
```

### Model Size Adjustments

**2B Parameter Variant:**
```yaml
# In config file
num_layers: 20
hidden_size: 1800
ffn_hidden_size: 4860
num_attention_heads: 14
```

**5B Parameter Variant:**
```yaml
# In config file
num_layers: 30
hidden_size: 2560
ffn_hidden_size: 6912
num_attention_heads: 20
```

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce `micro_batch_size` to 1
   - Increase `tensor_model_parallel_size`
   - Enable more aggressive activation checkpointing

2. **Slow Training:**
   - Verify binary dataset files are on fast storage (SSD/NVMe)
   - Increase `num_workers` for data loading
   - Check GPU utilization with `nvidia-smi`

3. **Dataset Not Found:**
   - Ensure preprocessing completed successfully
   - Check file paths in configuration match preprocessed files
   - Verify `.bin` and `.idx` files exist

### Performance Tuning

**For A100 80GB:**
```yaml
micro_batch_size: 4
tensor_model_parallel_size: 1
```

**For V100 32GB:**
```yaml
micro_batch_size: 1
tensor_model_parallel_size: 4
activations_checkpoint_num_layers: 4
```

## Next Steps

1. **Evaluation**: Use standard language modeling benchmarks
2. **Fine-tuning**: Convert to instruction-following with SFT
3. **Deployment**: Export to ONNX or TensorRT for inference
4. **Scaling**: Train larger variants with more data

## Support

- **NeMo Documentation**: https://docs.nvidia.com/deeplearning/nemo/
- **Intercom NeMo Fork**: https://github.com/intercom/NeMo
- **Qwen3-Next Paper**: [Architecture details and benchmarks]
- **Community**: Intercom NeMo fork discussions for technical issues

Happy training! 🚀