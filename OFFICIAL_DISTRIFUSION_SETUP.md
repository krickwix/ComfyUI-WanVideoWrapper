# Official DistriFusion Integration Guide

## 🎯 **Using MIT-HAN-Lab's Official DistriFusion**

This guide covers the integration of the **official DistriFusion implementation** from [MIT-HAN-Lab](https://github.com/mit-han-lab/distrifuser) with ComfyUI WanVideoWrapper.

## 📦 **Installation**

### 1. Install Official DistriFusion
```bash
pip install git+https://github.com/mit-han-lab/distrifuser.git
```

### 2. Install WanVideoWrapper Dependencies
```bash
cd ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
pip install -r requirements.txt
```

### 3. Fix HuggingFace Hub Dependency (if needed)
```bash
pip install huggingface_hub<0.20.0 --force-reinstall
```

## 🎮 **Available Nodes**

Look for these nodes under **`WanVideoWrapper/Official DistriFusion`**:

### 🎯 **Official DistriFusion Model Loader**
- **Purpose**: Load WanVideo models with official DistriFusion distribution
- **Features**: Uses MIT-HAN-Lab's research implementation
- **Best for**: Production use with proven research implementation

### 🎯 **Official DistriFusion Sampler**
- **Purpose**: Sample using official DistriFusion distributed inference
- **Features**: Proper patch parallelism and boundary communication
- **Best for**: High-quality distributed video generation

### 🎯 **Official DistriFusion Status**
- **Purpose**: Monitor official DistriFusion performance and configuration
- **Features**: Real-time status, GPU information, model configuration
- **Best for**: Debugging and performance monitoring

## 🚀 **Quick Start**

### Step 1: Load Model
1. Add **"🎯 Official DistriFusion Model Loader"** node
2. Select your WanVideo model
3. Configure distribution settings:
   - **GPUs**: Number of GPUs to use
   - **Patch Size**: Size of patches (default: 64)
   - **Split Mode**: How to split across GPUs (spatial/temporal/spatiotemporal)
4. Connect to sampling workflow

### Step 2: Sample
1. Add **"🎯 Official DistriFusion Sampler"** node
2. Connect the model from Step 1
3. Configure sampling parameters
4. Run the workflow

## ⚙️ **Configuration Options**

### Model Loader Settings
- **Precision**: fp16, bf16, fp32
- **Quantization**: disabled, fp8_e4m3fn, fp8_e5m2
- **GPUs**: 1-8 GPUs for distribution
- **Patch Size**: 32-128 (step 8)
- **Patch Overlap**: 0-32 pixels
- **Split Mode**: spatial, temporal, spatiotemporal
- **World Size**: Total processes (usually same as GPUs)
- **Rank**: Process rank (0 for single process)

### Sampler Settings
- **Sync Frequency**: How often to synchronize between GPUs
- **Standard sampling parameters**: steps, cfg, sampler, scheduler, etc.

## 🔧 **Advanced Usage**

### Multi-GPU Setup
```python
# Example configuration for 4 GPUs
num_gpus = 4
patch_size = 64
patch_overlap = 8
split_mode = "spatial"  # or "temporal", "spatiotemporal"
world_size = 4
rank = 0  # Set different rank for each process
```

### Performance Optimization
- **Patch Size**: Larger patches = less communication overhead
- **Patch Overlap**: Smaller overlap = faster but may have artifacts
- **Split Mode**: 
  - **Spatial**: Best for high-resolution videos
  - **Temporal**: Best for long videos
  - **Spatiotemporal**: Best for both high-res and long videos

## 📊 **Performance Monitoring**

Use the **"🎯 Official DistriFusion Status"** node to monitor:
- ✅ DistriFusion availability
- 📊 GPU configuration
- 🔲 Patch settings
- 🔄 Current inference step
- 🎮 CUDA device information

## 🔍 **Troubleshooting**

### Official DistriFusion Not Available
```
⚠️ Official DistriFusion not available: [...]
   Install with: pip install git+https://github.com/mit-han-lab/distrifuser.git
```

**Solution**: Install the official package
```bash
pip install git+https://github.com/mit-han-lab/distrifuser.git
```

### Import Errors
If you see import errors, check:
1. **PyTorch installation**: `python -c "import torch; print(torch.__version__)"`
2. **CUDA availability**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **DistriFusion installation**: `python -c "import distrifuser; print(distrifuser.__version__)"`

### GPU Memory Issues
- Reduce **patch size** (32 instead of 64)
- Reduce **patch overlap** (4 instead of 8)
- Use **quantization** (fp8_e4m3fn)
- Reduce **number of GPUs**

## 🆚 **Custom vs Official DistriFusion**

| Feature | Custom DistriFusion | Official DistriFusion |
|---------|-------------------|----------------------|
| **Implementation** | Custom wrapper | MIT-HAN-Lab research |
| **Stability** | Experimental | Production-ready |
| **Features** | Basic functionality | Full research features |
| **Performance** | Good | Optimized |
| **Documentation** | Limited | Research paper |
| **Support** | Community | Research team |

## 📚 **Research Reference**

The official DistriFusion implementation is based on:
- **Paper**: "DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models"
- **Authors**: MIT-HAN-Lab
- **Conference**: CVPR 2024
- **Repository**: https://github.com/mit-han-lab/distrifuser

## 🎉 **Benefits of Official Implementation**

✅ **Research Proven**: Based on peer-reviewed research  
✅ **Optimized**: Fine-tuned for performance  
✅ **Maintained**: Active development by research team  
✅ **Compatible**: Works with standard diffusion models  
✅ **Scalable**: Tested on multiple GPU configurations  

---

**For production use, we recommend the Official DistriFusion implementation for its stability, performance, and research backing.** 