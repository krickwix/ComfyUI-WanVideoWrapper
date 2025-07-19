# DistriFusion ComfyUI Setup Guide

## ✅ **ISSUE RESOLVED**

The DistriFusion nodes should now appear in your ComfyUI interface! The import issues have been comprehensively fixed.

## 🎯 **Quick Check**

Run this verification script in your ComfyUI environment:
```bash
cd ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
python verify_distrifusion_nodes.py
```

## 📍 **Where to Find DistriFusion Nodes**

In ComfyUI, look for these nodes under **`WanVideoWrapper/DistriFusion`** category:

### 🆕 **Recommended Nodes** (New Streamlined Interface)
- **🎯 DistriFusion Model Loader** - Auto-configuring loader with smart defaults
- **⚙️ DistriFusion Distribution Config** - Configuration presets for different setups

### 📦 **Legacy Nodes** (Original Implementation)  
- **📦 DistriFusion WanVideo Model Loader (Legacy)** - Original detailed interface
- **🎯 DistriFusion WanVideo Sampler** - Distributed sampling
- **🔧 DistriFusion Setup** - Environment setup utilities
- **📊 DistriFusion Status** - Performance monitoring

## 🚀 **Getting Started**

### Option 1: **Quick Start** (Recommended)
1. Add `DistriFusion Model Loader` node
2. Select your WanVideo model 
3. Choose GPU count (auto-detected)
4. Set split strategy (auto-selected based on model)
5. Connect to your sampling workflow

### Option 2: **Advanced Setup**
1. Use `DistriFusion Distribution Config` for custom settings
2. Connect to `DistriFusion WanVideo Model Loader (Legacy)`
3. Use `DistriFusion Setup` for environment configuration
4. Monitor with `DistriFusion Status`

## 🔧 **Requirements**

For DistriFusion nodes to appear, you need:

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers

# Optional but recommended
pip install xformers  # For memory efficiency
```

## ⚠️ **Troubleshooting**

### Nodes Don't Appear
1. **Check ComfyUI Console** for import error messages
2. **Verify Dependencies**: Run the verification script above
3. **Restart ComfyUI** completely after installing dependencies
4. **Check CUDA**: Ensure PyTorch detects your GPUs with `torch.cuda.is_available()`

### Import Warnings
The following warnings are **NORMAL** if dependencies are missing:
```
Warning: Could not import DistriFusion wrapper: No module named 'torch'
Warning: Could not import patch manager: [...]
⚠️ DistriFusion not available: [...]
```

These indicate DistriFusion is properly detecting missing dependencies and will enable automatically once you install them.

### Clear Error Messages
DistriFusion now provides clear feedback:
- ✅ **Success**: "DistriFusion nodes loaded successfully"  
- ⚠️ **Missing deps**: Clear instructions on what to install
- ❌ **Import errors**: Specific module/function that failed

## 🎉 **What's Fixed**

✅ **Import Path Issues** - Works with ComfyUI's module structure  
✅ **Silent Failures** - Clear error messages and warnings  
✅ **Graceful Degradation** - Other WanVideo nodes work even if DistriFusion fails  
✅ **Dependency Detection** - Auto-detects and reports missing requirements  
✅ **Multiple Import Contexts** - Works regardless of how ComfyUI loads the module  

## 🔗 **Repository Status**

- **Branch**: `krickwix/distrifusion` 
- **Latest Commit**: `6fafdb0`
- **Status**: ✅ Ready for use

## 📋 **Node Descriptions**

### DistriFusion Model Loader
- **Purpose**: Streamlined model loading with auto-configuration
- **Features**: GPU detection, strategy auto-selection, memory optimization
- **Best for**: Most users who want simple setup

### DistriFusion Distribution Config  
- **Purpose**: Reusable configuration presets
- **Features**: Save/load distribution settings, multiple GPU profiles
- **Best for**: Users with multiple setups or complex configurations

### Legacy Nodes
- **Purpose**: Detailed control over every aspect of distributed inference
- **Features**: Fine-grained control, advanced monitoring, custom setups
- **Best for**: Advanced users who need maximum control

---

🎊 **The DistriFusion nodes should now appear in your ComfyUI interface!**

If you still don't see them after installing dependencies and restarting ComfyUI, please check the console output for specific error messages. 