# DistriFusion for WanVideo Models

DistriFusion implementation for distributed parallel inference of WanVideo models across multiple GPUs using displaced patch parallelism.

## Overview

DistriFusion enables efficient multi-GPU inference for WanVideo models by:

1. **Displaced Patch Parallelism**: Splits video tensors into patches distributed across GPUs
2. **Asynchronous Communication**: Uses cached activations from previous steps to reduce communication overhead
3. **Boundary Synchronization**: Maintains patch interactions for seamless video generation
4. **Memory Optimization**: Reduces per-GPU memory requirements through distributed processing

## Performance Benefits

- **Up to 6.1x speedup** on 8 GPUs for high-resolution video generation
- **Reduced memory usage** per GPU through patch distribution
- **Efficient communication** with asynchronous boundary updates
- **Scalable** to multiple GPUs with minimal communication overhead

## Installation

DistriFusion is included with the WanVideoWrapper. Ensure you have:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers
```

For multi-GPU support, ensure NCCL is properly installed:
```bash
# NCCL is typically included with PyTorch CUDA installations
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

## Quick Start

### 1. Basic Setup

```python
from ComfyUI_WanVideoWrapper.distrifusion import (
    DistriFusionWanVideoModelLoader,
    DistriFusionWanVideoSampler
)

# Load model with DistriFusion
model_loader = DistriFusionWanVideoModelLoader()
distrifusion_model = model_loader.load_distrifusion_model(
    model="your_wanvideo_model.safetensors",
    enable_distrifusion=True,
    num_gpus=2,
    split_mode="spatial",
    patch_overlap=8,
    process_rank=0  # Set by launcher
)

# Sample with distributed inference
sampler = DistriFusionWanVideoSampler()
result = sampler.sample_distrifusion(
    distrifusion_model=distrifusion_model,
    positive=positive_conditioning,
    negative=negative_conditioning,
    latent_image=latent,
    steps=20,
    cfg=8.0
)
```

### 2. Using the Launcher

For automatic multi-GPU setup:

```bash
# Launch with 4 GPUs
python -m ComfyUI_WanVideoWrapper.distrifusion.launcher \
    --script your_inference_script.py \
    --gpus 4 \
    --backend nccl

# Create script from ComfyUI workflow
python -m ComfyUI_WanVideoWrapper.distrifusion.launcher \
    --create-script \
    --workflow your_workflow.json
```

### 3. ComfyUI Node Usage

In ComfyUI, use the DistriFusion nodes:

1. **DistriFusion WanVideo Model Loader**
   - Enable DistriFusion: ✓
   - Number of GPUs: 2-8
   - Split Mode: spatial/temporal/spatiotemporal
   - Patch Overlap: 8-32

2. **DistriFusion WanVideo Sampler**
   - Connect DistriFusion model
   - Set async communication: ✓
   - Sync frequency: 1-10

## Configuration Options

### Split Modes

- **Spatial**: Split video frames spatially across GPUs (recommended)
- **Temporal**: Split video temporally across GPUs
- **Spatiotemporal**: Combine both spatial and temporal splitting

### Patch Parameters

- **Patch Overlap**: Boundary size for inter-patch communication (8-32 pixels)
- **Warmup Steps**: Number of initial steps with full synchronization (4-10)
- **Async Updates**: Enable asynchronous boundary updates for better performance

### Memory Optimization

- **Load Device**: Choose main_device for speed or offload_device for memory
- **Quantization**: Use fp8 quantization to reduce memory usage
- **Block Swapping**: Enable block swapping for large models

## Advanced Usage

### Custom Distributed Setup

```python
import torch.distributed as dist
from ComfyUI_WanVideoWrapper.distrifusion import create_distrifusion_model

# Manual distributed initialization
dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size
)

# Create DistriFusion model
wan_model = load_your_wan_model()
distrifusion_model = create_distrifusion_model(
    wan_model=wan_model,
    num_devices=world_size,
    split_mode="spatial",
    world_size=world_size,
    rank=rank
)

# Process with step tracking
for step in range(num_steps):
    distrifusion_model.update_step(step)
    output = distrifusion_model(input_tensor, timestep, context)
    distrifusion_model.synchronize()
```

### Performance Tuning

```python
# Optimize for your hardware
patch_config = PatchConfig(
    num_devices=num_gpus,
    split_mode=PatchSplitMode.SPATIAL_ONLY,
    patch_overlap=16,  # Increase for better quality
    warmup_steps=2,    # Decrease for speed
    async_boundary_update=True,
    sync_first_step=True
)
```

## Troubleshooting

### Common Issues

1. **NCCL Initialization Errors**
   ```bash
   export NCCL_DEBUG=INFO
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

2. **Memory Issues**
   - Reduce patch overlap
   - Enable quantization
   - Use offload_device for model loading

3. **Communication Timeouts**
   - Check network connectivity between GPUs
   - Reduce world_size
   - Use gloo backend for debugging

### Performance Monitoring

```python
# Check DistriFusion status
status_node = DistriFusionStatus()
status = status_node.get_status(distrifusion_model)
print(status)
```

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable distributed debugging
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_DEBUG'] = 'INFO'
```

## Best Practices

### For Optimal Performance

1. **Use spatial splitting** for most workloads
2. **Start with 2-4 GPUs** before scaling up
3. **Enable async communication** after warmup
4. **Monitor memory usage** across all GPUs
5. **Use NCCL backend** for multi-GPU systems

### For Quality

1. **Increase patch overlap** for seamless boundaries
2. **Use more warmup steps** for stability
3. **Enable synchronous mode** for critical applications
4. **Test with single GPU first** to establish baseline

### For Memory Efficiency

1. **Use temporal splitting** for long videos
2. **Enable quantization** when possible
3. **Reduce batch size** and increase GPU count
4. **Use gradient checkpointing** if available

## Example Workflows

### High-Resolution Video Generation

```python
# 4K video generation with 4 GPUs
distrifusion_model = model_loader.load_distrifusion_model(
    model="wan_video_4k.safetensors",
    num_gpus=4,
    split_mode="spatial",
    patch_overlap=16,
    warmup_steps=4
)
```

### Long Video Generation

```python
# Long video with temporal splitting
distrifusion_model = model_loader.load_distrifusion_model(
    model="wan_video_long.safetensors",
    num_gpus=8,
    split_mode="temporal",
    temporal_chunk_size=32,
    patch_overlap=4
)
```

### Memory-Constrained Setup

```python
# 2x 16GB GPUs
distrifusion_model = model_loader.load_distrifusion_model(
    model="wan_video.safetensors",
    num_gpus=2,
    split_mode="spatial",
    quantization="fp8_e4m3fn",
    load_device="offload_device",
    patch_overlap=8
)
```

## API Reference

See individual module documentation:

- [`PatchManager`](patch_manager.py): Handles patch splitting and reconstruction
- [`AsyncPatchCommunicator`](communication.py): Manages inter-GPU communication
- [`DistriFusionWanModel`](distrifusion_wrapper.py): Main distributed model wrapper
- [`DistriFusionLauncher`](launcher.py): Multi-process launcher utilities

## Contributing

To contribute to DistriFusion:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Test with multiple GPU configurations

## License

Same as WanVideoWrapper - see main LICENSE file. 