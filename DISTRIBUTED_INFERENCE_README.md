# WanVideo Distributed Inference with Ulysses Distribution

This document explains how to use the new distributed inference nodes in ComfyUI-WanVideoWrapper to leverage multiple GPUs for faster Wan2.2-Lightning inference.

## 🚀 **Overview**

The distributed inference system provides multi-GPU support using the same Ulysses distribution mechanism that we successfully implemented in the standalone Wan2.2-Lightning setup. This allows you to:

- **Scale across multiple GPUs** (2-8 GPUs supported)
- **Use Ulysses distribution** for optimal memory sharing
- **Enable FSDP** for model sharding
- **Maintain the same quality** with faster inference
- **Integrate seamlessly** with existing ComfyUI workflows
- **Maintain backward compatibility** with existing workflows

## 📋 **Prerequisites**

1. **DeepSpeed**: Required for FSDP + Ulysses support
2. **Multiple GPUs**: At least 2 GPUs with sufficient VRAM
3. **Wan2.2-Lightning Models**: Base model and LoRA weights must be downloaded
4. **ComfyUI**: Latest version with WanVideoWrapper installed

## 🔧 **Installation**

### 1. Update Requirements
The `requirements.txt` has been updated to include DeepSpeed:
```bash
pip install -r requirements.txt
```

### 2. Restart ComfyUI
After installing DeepSpeed, restart ComfyUI to load the new nodes.

## 🎯 **Integrated Distributed Inference**

### **LoadWanVideoModel - Distributed Options**
The distributed inference configuration is now integrated directly into the `LoadWanVideoModel` node for seamless compatibility:

**Standard Parameters:**
- `model`: Model name (e.g., "Wan2.2-T2V-A14B")
- `base_precision`: Precision (bf16, fp16, fp32)
- `load_device`: Device for loading (main_device, offload_device)
- `quantization`: Quantization method

**Distributed Inference Options:**
- `enable_distributed`: Enable multi-GPU inference (default: False)
- `gpu_count`: Number of GPUs (2-8, default: 2)
- `use_ulysses`: Enable Ulysses distribution (default: True)
- `use_fsdp`: Enable FSDP model sharding (default: True)
- `master_port`: Port for distributed communication (default: 29501)

### **WanVideoDistributedInference**
Main distributed inference node that runs multi-GPU inference using the configuration from the loaded model.

**Inputs:**
- `model`: WanVideo model with distributed configuration
- `text_embeds`: Text embeddings from text encoder
- `prompt`: Text prompt for video generation
- `width/height`: Video dimensions (832x480 recommended)
- `num_frames`: Number of video frames (121 = 2 seconds at 60fps)
- `sample_steps`: Sampling steps (20 for Lightning)
- `offload_model`: Enable model offloading

**Outputs:**
- `video_path`: Path to generated video
- `log_output`: Inference log output
- `status`: Success/error status

## 🔄 **Workflow Example**

### **Basic Multi-GPU Workflow**
1. **Load WanVideo Model**: Use `LoadWanVideoModel` with distributed options enabled
2. **Load Text Encoder**: Use `LoadWanVideoT5TextEncoder` for text processing
3. **Encode Text**: Use `WanVideoTextEncode` to create text embeddings
4. **Run Inference**: Use `WanVideoDistributedInference` to generate video
5. **Save Video**: Use `SaveVideo` to save the generated video

### **Workflow Files**
- **Integrated Approach**: `example_workflows/distributed_inference_workflow_integrated.json` (recommended)
- **Legacy Approach**: `example_workflows/distributed_inference_workflow.json` (with separate config node)

## 🔄 **Backward Compatibility**

### **Existing Workflows**
- **All existing workflows continue to work** without modification
- **Distributed inference is disabled by default** (`enable_distributed=False`)
- **No breaking changes** to existing functionality

### **Enabling Distributed Inference**
To enable distributed inference in existing workflows:
1. **Set `enable_distributed=True`** in `LoadWanVideoModel`
2. **Configure GPU count and options** as needed
3. **Replace standard inference** with `WanVideoDistributedInference` node

### **Migration Path**
```
Before (Standard):
LoadWanVideoModel → WanVideoInference

After (Distributed):
LoadWanVideoModel (enable_distributed=True) → WanVideoDistributedInference
```

## ⚙️ **Configuration Tips**

### **Optimal Settings for Different GPU Counts**

#### **2 GPUs**
```json
{
  "gpu_count": 2,
  "use_ulysses": true,
  "use_fsdp": true,
  "offload_model": true,
  "width": 832,
  "height": 480,
  "sample_steps": 20
}
```

#### **4 GPUs**
```json
{
  "gpu_count": 4,
  "use_ulysses": true,
  "use_fsdp": true,
  "offload_model": true,
  "width": 1024,
  "height": 576,
  "sample_steps": 20
}
```

#### **8 GPUs**
```json
{
  "gpu_count": 8,
  "use_ulysses": true,
  "use_fsdp": true,
  "offload_model": true,
  "width": 1280,
  "height": 720,
  "sample_steps": 20
}
```

### **Memory Optimization**
- **Always enable** `offload_model` for multi-GPU setups
- **Use lower resolutions** (832x480) for higher GPU counts
- **Enable memory_optimization** in config
- **Use bf16 precision** for best performance/memory balance

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Port Conflicts**
If you get port conflicts, change the `master_port` in the config:
```json
{
  "master_port": 29502
}
```

#### **Out of Memory**
- Reduce resolution (try 832x480)
- Enable `offload_model`
- Reduce `num_frames`
- Use fewer GPUs

#### **FSDP Errors**
- Ensure DeepSpeed is installed
- Try disabling `use_fsdp` temporarily
- Check GPU compatibility

### **Debug Mode**
Enable detailed logging by setting the log level in your ComfyUI configuration.

## 📊 **Performance Expectations**

### **Speed Improvements**
- **2 GPUs**: ~1.5-2x faster than single GPU
- **4 GPUs**: ~3-4x faster than single GPU  
- **8 GPUs**: ~6-8x faster than single GPU

### **Memory Usage**
- **Per GPU**: ~20-30GB VRAM (depending on resolution)
- **Total**: Scales with GPU count
- **Model Offloading**: Reduces per-GPU memory usage

## 🔗 **Integration with Existing Workflows**

The distributed inference nodes are designed to be drop-in replacements for existing WanVideo workflows:

1. **Replace** `WanVideoInference` with `WanVideoDistributedInference`
2. **Add** `WanVideoDistributedConfig` for configuration
3. **Connect** the same model and text embedding inputs
4. **Adjust** parameters for multi-GPU optimization

## 🎬 **Example Prompts**

### **High-Quality Video Generation**
```
Prompt: "A futuristic cityscape with flying cars and neon lights at night, cinematic lighting, 8K quality"
Settings: 1280x720, 121 frames, 20 sample steps, 8 GPUs
```

### **Fast Prototyping**
```
Prompt: "A beautiful sunset over mountains, golden hour lighting"
Settings: 832x480, 60 frames, 20 sample steps, 2 GPUs
```

## 🏆 **Best Practices**

1. **Start Small**: Begin with 2 GPUs and scale up
2. **Monitor Memory**: Watch GPU memory usage during inference
3. **Use Optimal Resolution**: 832x480 for most use cases
4. **Enable Offloading**: Always use model offloading for multi-GPU
5. **Test Prompts**: Validate with shorter sequences first
6. **Backup Workflows**: Save your working configurations

## 🆘 **Support**

If you encounter issues:

1. Check the ComfyUI console for error messages
2. Verify DeepSpeed installation: `pip show deepspeed`
3. Test with single GPU first
4. Check GPU compatibility and drivers
5. Review the log output for specific errors

## 🚀 **Future Enhancements**

Planned improvements:
- **Automatic GPU detection** and configuration
- **Dynamic resolution scaling** based on available memory
- **Batch processing** across multiple GPUs
- **Real-time progress monitoring** in ComfyUI
- **Integration with ComfyUI's queue system**

---

**The distributed inference system brings the full power of multi-GPU Ulysses distribution directly into ComfyUI, enabling faster, more efficient Wan2.2-Lightning video generation! 🎬✨**
