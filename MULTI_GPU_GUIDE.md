# Multi-GPU Parallelism Guide for WanVideo

This guide explains how to use PyTorch parallelism when loading WanVideo models with multiple GPUs.

## Overview

The WanVideo wrapper supports three types of multi-GPU parallelism:

1. **DataParallel**: Simple parallelism that replicates the model across GPUs
2. **Block Distribution**: Distributes transformer blocks across different GPUs
3. **DistributedDataParallel**: Advanced distributed training (requires proper setup)

## Prerequisites

- Multiple CUDA-compatible GPUs
- PyTorch with CUDA support
- WanVideo models (14B, 1.3B, etc.)
- ComfyUI with WanVideo wrapper installed

## Installation

1. Install the required dependencies:
```bash
pip install GPUtil psutil
```

2. Ensure your ComfyUI installation includes the WanVideo wrapper

## Usage

### Method 1: Using the Multi-GPU Loader Node

The easiest way to use multi-GPU parallelism is through the **WanVideo Multi-GPU Loader** node:

1. **Add the node to your workflow**:
   - Search for "WanVideo Multi-GPU Loader" in the node browser
   - Add it to your workflow

2. **Configure the settings**:
   - **Model**: Select your WanVideo model
   - **Base Precision**: Choose precision (bf16 recommended)
   - **Load Device**: Choose initial loading device
   - **Parallelism Type**: Select the type of parallelism
   - **GPU IDs**: Enter comma-separated GPU IDs (e.g., "0,1,2,3")
   - **Enable Block Distribution**: Enable for block distribution mode

3. **Parallelism Types Explained**:

   **DataParallel** (`data_parallel`):
   - Replicates the entire model across all GPUs
   - Good for smaller models that fit in GPU memory
   - Automatically handles data distribution
   - Use when: Model size < Total GPU memory

   **Block Distribution** (`block_distribution`):
   - Distributes transformer blocks across different GPUs
   - Good for large models that don't fit in single GPU
   - Requires manual data movement between GPUs
   - Use when: Model size > Single GPU memory

   **Distributed** (`distributed`):
   - Advanced distributed training setup
   - Requires proper DDP initialization
   - Best for training scenarios
   - Use when: Advanced distributed training needed

### Method 2: Using the Utility Script

You can also use the provided utility script to automatically determine optimal settings:

```python
from multi_gpu_utils import setup_optimal_parallelism, validate_gpu_setup

# Get optimal configuration for your model
config = setup_optimal_parallelism("path/to/your/model.safetensors")
print(f"Recommended config: {config}")

# Validate your GPU setup
is_valid, message = validate_gpu_setup([0, 1, 2])
print(f"Setup valid: {is_valid}")
print(f"Message: {message}")
```

### Method 3: Manual Configuration

For advanced users, you can manually configure the parallelism:

```python
# In your ComfyUI workflow
{
    "parallelism_type": "block_distribution",
    "gpu_ids": "0,1,2,3",
    "enable_block_distribution": True
}
```

## Configuration Examples

### Example 1: Two RTX 4090s (24GB each) with WanVideo 14B

```json
{
    "model": "wanvideo_14b.safetensors",
    "base_precision": "bf16",
    "load_device": "main_device",
    "parallelism_type": "data_parallel",
    "gpu_ids": "0,1",
    "enable_block_distribution": false
}
```

**Reasoning**: 14B model fits comfortably in 48GB total memory, so DataParallel is optimal.

### Example 2: Four RTX 3080s (10GB each) with WanVideo 14B

```json
{
    "model": "wanvideo_14b.safetensors",
    "base_precision": "bf16",
    "load_device": "main_device",
    "parallelism_type": "block_distribution",
    "gpu_ids": "0,1,2,3",
    "enable_block_distribution": true
}
```

**Reasoning**: 14B model doesn't fit in single 10GB GPU, so block distribution is needed.

### Example 3: Single RTX 4090 with WanVideo 1.3B

```json
{
    "model": "wanvideo_1_3b.safetensors",
    "base_precision": "bf16",
    "load_device": "main_device",
    "parallelism_type": "none",
    "gpu_ids": "0",
    "enable_block_distribution": false
}
```

**Reasoning**: Single GPU is sufficient for 1.3B model.

## Performance Optimization Tips

### 1. Memory Management

- **Monitor GPU memory usage** during inference
- **Use appropriate precision** (bf16 for most cases)
- **Enable block swapping** for large models
- **Consider quantization** for memory-constrained setups

### 2. DataParallel Optimization

- **Batch size**: Increase batch size to utilize multiple GPUs effectively
- **Memory alignment**: Ensure data is properly aligned across GPUs
- **Communication overhead**: Minimize data transfer between GPUs

### 3. Block Distribution Optimization

- **Block placement**: Distribute blocks evenly across GPUs
- **Data movement**: Minimize cross-GPU data transfers
- **Memory balance**: Ensure balanced memory usage across GPUs

### 4. General Tips

- **Warm-up runs**: Run a few inference passes to warm up GPUs
- **Temperature monitoring**: Monitor GPU temperatures during long runs
- **Error handling**: Implement proper error handling for GPU failures

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable block swapping
   - Use lower precision
   - Enable block distribution

2. **GPU Communication Errors**:
   - Check GPU compatibility
   - Verify CUDA driver versions
   - Ensure proper GPU initialization

3. **Performance Issues**:
   - Monitor GPU utilization
   - Check for memory bottlenecks
   - Verify parallelism configuration

4. **Model Loading Failures**:
   - Check model file integrity
   - Verify GPU memory availability
   - Ensure proper model format

### Debugging Commands

```python
# Check GPU status
from multi_gpu_utils import print_gpu_status
print_gpu_status()

# Validate GPU setup
from multi_gpu_utils import validate_gpu_setup
is_valid, message = validate_gpu_setup([0, 1])
print(f"Valid: {is_valid}, Message: {message}")

# Get optimal configuration
from multi_gpu_utils import setup_optimal_parallelism
config = setup_optimal_parallelism("model.safetensors")
print(config)
```

## Advanced Usage

### Custom Block Distribution

For advanced users, you can customize block distribution:

```python
# Custom block distribution logic
def custom_block_distribution(model, gpu_ids):
    num_blocks = len(model.blocks)
    num_gpus = len(gpu_ids)
    
    # Custom distribution strategy
    for i, block in enumerate(model.blocks):
        target_gpu = gpu_ids[i % num_gpus]  # Round-robin distribution
        block.to(f'cuda:{target_gpu}')
```

### Memory Monitoring

Monitor memory usage during inference:

```python
from multi_gpu_utils import monitor_memory_usage

# Start monitoring
monitor_thread = monitor_memory_usage(model, interval=5.0, duration=60.0)

# Your inference code here
# ...

# Monitoring will automatically stop after duration
```

### Performance Benchmarking

Benchmark different configurations:

```python
from multi_gpu_utils import benchmark_multi_gpu_performance

# Benchmark performance
results = benchmark_multi_gpu_performance(model, input_data, num_runs=5)
print(f"Benchmark results: {results}")
```

## Best Practices

1. **Start Simple**: Begin with DataParallel for smaller models
2. **Monitor Resources**: Always monitor GPU memory and temperature
3. **Test Thoroughly**: Test your configuration with sample data
4. **Document Settings**: Keep track of optimal settings for your hardware
5. **Update Regularly**: Keep drivers and PyTorch updated
6. **Backup Configurations**: Save working configurations for future use

## Hardware Recommendations

### Minimum Requirements

- **GPU Memory**: 8GB per GPU (minimum)
- **GPU Count**: 2+ GPUs for meaningful parallelism
- **PCIe**: PCIe 3.0 x16 or better for multi-GPU communication
- **Power**: Sufficient power supply for multiple GPUs

### Recommended Configurations

- **Entry Level**: 2x RTX 3080 (10GB each)
- **Mid Range**: 2x RTX 4080 (16GB each)
- **High End**: 2x RTX 4090 (24GB each)
- **Professional**: 4x A100 (40GB each)

## Conclusion

Multi-GPU parallelism can significantly improve WanVideo model performance and enable the use of larger models. Start with the automatic configuration tools and gradually optimize based on your specific hardware and use case.

For additional support, refer to:
- PyTorch documentation on DataParallel and DistributedDataParallel
- WanVideo wrapper documentation
- ComfyUI community forums

Happy multi-GPU video generation! 