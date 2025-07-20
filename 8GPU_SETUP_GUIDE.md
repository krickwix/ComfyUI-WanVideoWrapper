# 🚀 8-GPU Distributed Inference Setup Guide

## Overview
This guide helps you set up Wan2.1 distributed inference to utilize all 8 GPUs for maximum performance.

## 🎯 Quick Start

### 1. Launch ComfyUI with 8-GPU Support
```bash
# From the ComfyUI-WanVideoWrapper directory
./launch_8gpu_comfyui.sh
```

### 2. Load the 8-GPU Workflow
- Open ComfyUI in your browser
- Load: `example_workflows/8gpu_distributed_workflow.json`

### 3. Update Model Paths
Make sure these models are in the correct locations:
- **Main Model**: `models/diffusion_models/wan2.1_14b.safetensors`
- **T5 Encoder**: `models/text_encoders/umt5_xxl_fp16.safetensors`
- **VAE**: `models/vae/wan_vae.safetensors`

## ⚙️ 8-GPU Configuration

### Distributed Settings
```json
{
  "world_size": 8,           // Use all 8 GPUs
  "rank": 0,                 // Main process rank
  "use_fsdp": true,          // Enable FSDP for model sharding
  "use_context_parallel": true,  // Enable context parallel
  "ulysses_size": 2,         // 2x4 = 8 GPUs
  "ring_size": 4,            // 2x4 = 8 GPUs
  "t5_fsdp": true,           // Shard T5 encoder across GPUs
  "dit_fsdp": true           // Shard DIT model across GPUs
}
```

### Performance Optimizations
- **VRAM Management**: 50% offload for balanced memory usage
- **Block Swap**: 10 blocks for efficient memory management
- **Torch Compile**: Enabled for faster inference
- **TEA Cache**: Enabled for memory optimization

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Set Environment Variables
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8
export RANK=0
export NCCL_DEBUG=INFO
```

### 2. Launch ComfyUI
```bash
cd /path/to/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### 3. Configure in ComfyUI
1. Add `Wan2.1 Distributed Config` node
2. Set `world_size` to 8
3. Enable `use_fsdp` and `use_context_parallel`
4. Set `ulysses_size=2` and `ring_size=4`

## 📊 Expected Performance

### Memory Distribution
- **Model Parameters**: Sharded across 8 GPUs using FSDP
- **Context Parallel**: Sequence parallelism for long contexts
- **Memory Usage**: ~2-3GB per GPU for 14B model

### Speed Improvements
- **Inference Speed**: 3-5x faster than single GPU
- **Memory Efficiency**: Better memory utilization
- **Scalability**: Linear scaling with GPU count

## 🐛 Troubleshooting

### GPU Not Utilized
1. **Check Environment Variables**:
   ```bash
   echo $MASTER_ADDR $MASTER_PORT $WORLD_SIZE
   ```

2. **Verify GPU Visibility**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Check Distributed Initialization**:
   - Look for "Initialized distributed environment" in logs
   - Verify "Setup complete" message

### Memory Issues
1. **Reduce VRAM Management**: Set `offload_percent` to 0.3
2. **Increase Block Swap**: Set `blocks_to_swap` to 15
3. **Disable Context Parallel**: Set `use_context_parallel` to false

### Performance Issues
1. **Enable Torch Compile**: Set `backend` to "inductor"
2. **Optimize Attention**: Use "flash_attn_2" or "flash_attn_3"
3. **Adjust Batch Size**: Reduce if memory is constrained

## 🔍 Monitoring

### GPU Utilization
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check distributed processes
ps aux | grep python
```

### Log Messages
Look for these success messages:
- ✅ "Initialized distributed environment with 8 processes"
- ✅ "Applied FSDP to model"
- ✅ "Setup complete: FSDP=true, Context Parallel=true"

## 📝 Advanced Configuration

### Custom GPU Mapping
```python
# In the distributed config
"ulysses_size": 2,  # 2 groups
"ring_size": 4,     # 4 GPUs per group
# Total: 2 x 4 = 8 GPUs
```

### Memory Optimization
```python
# VRAM Management
"offload_percent": 0.5,  # 50% offload

# Block Swap
"blocks_to_swap": 10,    # Swap 10 blocks

# TEA Cache
"cache_device": "offload_device"
```

## 🎉 Success Indicators

When working correctly, you should see:
1. **All 8 GPUs active** in `nvidia-smi`
2. **Distributed initialization** messages in logs
3. **Faster inference** compared to single GPU
4. **Balanced memory usage** across GPUs

## 📞 Support

If you encounter issues:
1. Check the logs for error messages
2. Verify all environment variables are set
3. Ensure model files are in correct locations
4. Try reducing `world_size` to 4 for testing

---

**Happy 8-GPU distributed inference! 🚀** 