# XDit+USP Distributed Inference for WanVideo

This extension adds support for multi-GPU distributed inference using xdit+USP (Ultra-Scalable Parallelism) for WanVideo models in ComfyUI.

## Overview

The xdit+USP integration enables efficient distributed inference across multiple GPUs, significantly reducing memory usage per GPU and enabling inference of large models that wouldn't fit on a single GPU.

### Key Features

- **Multi-GPU Support**: Distribute model across 2-8 GPUs
- **Memory Efficiency**: Reduce per-GPU memory usage by 50-80%
- **Pipeline Parallelism**: Split model layers across GPUs
- **Tensor Parallelism**: Split attention heads and MLP layers
- **Data Parallelism**: Process multiple batches in parallel
- **Mixed Precision**: Support for FP16 and BF16 inference
- **Activation Checkpointing**: Memory optimization technique
- **Flash Attention**: Optimized attention computation
- **SDPA Support**: Scaled Dot-Product Attention

## Installation

### Prerequisites

- CUDA 11.8+ or ROCm 5.0+
- PyTorch 2.0+
- Multiple GPUs (2-8 recommended)
- Sufficient system memory (32GB+ recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install xdit>=0.1.0 usp>=0.1.0 torch>=2.0.0 transformers>=4.30.0
```

## Usage

### 1. Configure XDit+USP Settings

Use the **XDit+USP Config** node to configure distributed inference settings:

- **num_gpus**: Number of GPUs to use (1-8)
- **gpu_memory_fraction**: Fraction of GPU memory to use per GPU (0.1-1.0)
- **pipeline_parallel_size**: Pipeline parallel size (1-4)
- **tensor_parallel_size**: Tensor parallel size (1-4)
- **data_parallel_size**: Data parallel size (1-4)
- **use_fp16**: Enable FP16 precision
- **use_bf16**: Enable BF16 precision
- **enable_activation_checkpointing**: Enable memory optimization
- **max_batch_size**: Maximum batch size for inference
- **max_sequence_length**: Maximum sequence length
- **use_flash_attention**: Enable Flash Attention
- **use_sdpa**: Enable SDPA

### 2. Load Model with Distributed Inference

Use the **XDit+USP WanVideo Model Loader** node:

1. Select your WanVideo model
2. Choose base precision (fp32, bf16, fp16)
3. Connect the XDit+USP config
4. Optionally configure attention mode and LoRA

### Example Workflow

```
XDit+USP Config
├── num_gpus: 4
├── gpu_memory_fraction: 0.8
├── pipeline_parallel_size: 2
├── tensor_parallel_size: 2
└── use_fp16: True
    ↓
XDit+USP WanVideo Model Loader
├── model: wan2.1_14b.safetensors
├── base_precision: bf16
├── xdit_usp_config: [from config node]
└── attention_mode: sdpa
    ↓
[Your inference nodes]
```

## Configuration Guidelines

### GPU Memory Requirements

| Model Size | Single GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------------|------------|--------|--------|--------|
| 1.3B       | 8GB        | 4GB    | 2GB    | 1GB    |
| 14B        | 48GB       | 24GB   | 12GB   | 6GB    |

### Recommended Configurations

#### For 2 GPUs (24GB each)
```python
{
    "num_gpus": 2,
    "gpu_memory_fraction": 0.9,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 1,
    "data_parallel_size": 1,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

#### For 4 GPUs (24GB each)
```python
{
    "num_gpus": 4,
    "gpu_memory_fraction": 0.8,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 2,
    "data_parallel_size": 1,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

#### For 8 GPUs (24GB each)
```python
{
    "num_gpus": 8,
    "gpu_memory_fraction": 0.7,
    "pipeline_parallel_size": 4,
    "tensor_parallel_size": 2,
    "data_parallel_size": 1,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

## Performance Optimization

### Memory Optimization

1. **Enable Activation Checkpointing**: Reduces memory usage by recomputing activations
2. **Use Mixed Precision**: FP16/BF16 reduces memory usage by 50%
3. **Adjust GPU Memory Fraction**: Lower values use less memory per GPU
4. **Optimize Parallelism**: Balance pipeline, tensor, and data parallelism

### Speed Optimization

1. **Use Flash Attention**: Faster attention computation
2. **Enable SDPA**: Optimized attention implementation
3. **Overlap Communication**: Enable peer-to-peer communication overlap
4. **Optimize Batch Size**: Larger batches improve throughput

### Communication Optimization

1. **Pipeline Parallelism**: Reduces communication overhead
2. **Tensor Parallelism**: Efficient for attention-heavy models
3. **Data Parallelism**: Good for batch processing
4. **Overlap P2P Comm**: Overlaps communication with computation

## Troubleshooting

### Common Issues

#### "Not enough GPUs available"
- Check `torch.cuda.device_count()` to verify available GPUs
- Reduce `num_gpus` in configuration

#### "Out of memory"
- Reduce `gpu_memory_fraction`
- Enable `enable_activation_checkpointing`
- Use mixed precision (`use_fp16` or `use_bf16`)
- Increase parallelism (pipeline/tensor parallel)

#### "ImportError: xdit not available"
- Install xdit: `pip install xdit>=0.1.0`
- Check CUDA compatibility

#### "ImportError: usp not available"
- Install usp: `pip install usp>=0.1.0`
- Verify PyTorch version compatibility

### Performance Debugging

1. **Monitor GPU Memory**: Use `nvidia-smi` to check memory usage
2. **Check Communication**: Monitor inter-GPU communication
3. **Profile Inference**: Use PyTorch profiler for bottlenecks
4. **Validate Configuration**: Ensure parallelism settings are valid

## Advanced Configuration

### Custom Parallelism Strategies

```python
# Pipeline-only parallelism
{
    "pipeline_parallel_size": 4,
    "tensor_parallel_size": 1,
    "data_parallel_size": 1
}

# Tensor-only parallelism
{
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": 4,
    "data_parallel_size": 1
}

# Hybrid parallelism
{
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 2,
    "data_parallel_size": 1
}
```

### Memory-Efficient Settings

```python
{
    "gpu_memory_fraction": 0.6,
    "enable_activation_checkpointing": True,
    "enable_gradient_checkpointing": True,
    "use_fp16": True,
    "max_batch_size": 1
}
```

### High-Performance Settings

```python
{
    "gpu_memory_fraction": 0.95,
    "enable_activation_checkpointing": False,
    "use_fp16": True,
    "use_flash_attention": True,
    "use_sdpa": True,
    "overlap_p2p_comm": True,
    "max_batch_size": 4
}
```

## Limitations

1. **LoRA Support**: Experimental, may not work with all LoRA types
2. **VACE Models**: Limited support for VACE in distributed mode
3. **Model Compatibility**: Only tested with WanVideo models
4. **Memory Overhead**: Distributed inference has communication overhead
5. **Setup Complexity**: Requires proper GPU configuration

## Future Enhancements

- [ ] Full LoRA support for distributed inference
- [ ] VACE model optimization
- [ ] Dynamic parallelism adjustment
- [ ] Automatic memory optimization
- [ ] Support for other model architectures
- [ ] Integration with other distributed frameworks

## Contributing

Contributions are welcome! Please:

1. Test with different GPU configurations
2. Report performance issues
3. Suggest optimization strategies
4. Add support for new features

## License

This extension follows the same license as the main ComfyUI-WanVideoWrapper project. 