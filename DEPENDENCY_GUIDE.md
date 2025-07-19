# Dependency Management Guide

## 📦 **Requirements Files Overview**

WanVideoWrapper provides multiple requirements files to accommodate different use cases:

### 🎯 **requirements.txt** (Default)
- **Purpose**: Default requirements with official DistriFusion support
- **diffusers**: `==0.24.0` (required for official DistriFusion)
- **Use when**: You want official DistriFusion + all WanVideo features

### 🎯 **requirements_official_distrifusion.txt** (Recommended for DistriFusion)
- **Purpose**: Dedicated file for official DistriFusion
- **diffusers**: `==0.24.0` (exact version required)
- **Use when**: You specifically want official DistriFusion

### 🎯 **requirements_compatible.txt** (For newer diffusers)
- **Purpose**: Compatible with newer diffusers versions
- **diffusers**: `>=0.33.0` (newer versions)
- **Use when**: You want newer diffusers features (no official DistriFusion)

## 🔧 **Installation Options**

### Option 1: Official DistriFusion (Recommended)
```bash
cd ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
pip install -r requirements_official_distrifusion.txt
```

### Option 2: Newer Diffusers (No Official DistriFusion)
```bash
cd ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
pip install -r requirements_compatible.txt
```

### Option 3: Default (Official DistriFusion)
```bash
cd ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
pip install -r requirements.txt
```

## ⚠️ **Dependency Conflicts**

### The Problem
Official DistriFusion requires `diffusers==0.24.0`, but newer WanVideo features work with `diffusers>=0.33.0`.

### The Solution
Choose the appropriate requirements file based on your needs:

| Use Case | Requirements File | diffusers Version | Official DistriFusion |
|----------|------------------|-------------------|----------------------|
| **Production with DistriFusion** | `requirements_official_distrifusion.txt` | 0.24.0 | ✅ Available |
| **Latest features** | `requirements_compatible.txt` | ≥0.33.0 | ❌ Not available |
| **Default setup** | `requirements.txt` | 0.24.0 | ✅ Available |

## 🔍 **Troubleshooting**

### Error: "Cannot install -r requirements.txt and diffusers>=0.33.0"
**Cause**: Dependency conflict between official DistriFusion and newer diffusers

**Solution**:
```bash
# Choose one of these options:

# Option A: Use official DistriFusion (recommended)
pip install -r requirements_official_distrifusion.txt

# Option B: Use newer diffusers (no official DistriFusion)
pip install -r requirements_compatible.txt
```

### Error: "distrifuser 0.0.1b1 depends on diffusers==0.24.0"
**Cause**: Official DistriFusion has strict version requirements

**Solution**:
```bash
# Downgrade diffusers to compatible version
pip install diffusers==0.24.0 --force-reinstall
pip install git+https://github.com/mit-han-lab/distrifuser.git
```

## 📋 **Feature Comparison**

| Feature | Official DistriFusion | Custom DistriFusion | No DistriFusion |
|---------|----------------------|-------------------|-----------------|
| **Official DistriFusion** | ✅ Full support | ❌ Not available | ❌ Not available |
| **Custom DistriFusion** | ✅ Available | ✅ Full support | ❌ Not available |
| **Newer diffusers features** | ❌ Limited | ✅ Available | ✅ Full support |
| **Stability** | ✅ Research proven | ⚠️ Experimental | ✅ Stable |
| **Performance** | ✅ Optimized | ✅ Good | ✅ Standard |

## 🚀 **Quick Decision Guide**

### Choose Official DistriFusion if:
- ✅ You want research-proven distributed inference
- ✅ You have multiple GPUs
- ✅ You need production-ready performance
- ✅ You're okay with diffusers 0.24.0

### Choose Newer Diffusers if:
- ✅ You need latest diffusers features
- ✅ You don't need distributed inference
- ✅ You want maximum compatibility
- ✅ You prefer single-GPU operation

### Choose Custom DistriFusion if:
- ✅ You want distributed inference
- ✅ You need newer diffusers features
- ✅ You're okay with experimental implementation

## 📚 **Migration Guide**

### From Newer Diffusers to Official DistriFusion
```bash
# 1. Uninstall current packages
pip uninstall diffusers distrifuser -y

# 2. Install official DistriFusion requirements
pip install -r requirements_official_distrifusion.txt

# 3. Restart ComfyUI
```

### From Official DistriFusion to Newer Diffusers
```bash
# 1. Uninstall official DistriFusion
pip uninstall distrifuser -y

# 2. Upgrade diffusers
pip install diffusers>=0.33.0 --force-reinstall

# 3. Restart ComfyUI
```

---

**Recommendation**: For production use with multiple GPUs, use `requirements_official_distrifusion.txt` for the best performance and stability. 