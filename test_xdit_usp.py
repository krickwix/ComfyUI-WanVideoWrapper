#!/usr/bin/env python3
"""
Test script for XDit+USP distributed inference functionality
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_xdit_usp_imports():
    """Test if xdit and usp can be imported"""
    print("Testing XDit+USP imports...")
    
    try:
        from nodes_xdit_usp_loader import XDitUSPConfig, XDitUSPWanVideoModel
        print("✓ Successfully imported XDit+USP classes")
        return True
    except ImportError as e:
        print(f"✗ Failed to import XDit+USP classes: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return gpu_count >= 2

def test_xdit_usp_config():
    """Test XDit+USP configuration"""
    print("\nTesting XDit+USP configuration...")
    
    try:
        from nodes_xdit_usp_loader import XDitUSPConfig
        
        # Test default configuration
        config = XDitUSPConfig()
        print("✓ Created default XDit+USP config")
        
        # Test custom configuration
        config = XDitUSPConfig(
            num_gpus=2,
            gpu_memory_fraction=0.8,
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            use_fp16=True,
            enable_activation_checkpointing=True
        )
        print("✓ Created custom XDit+USP config")
        
        # Print configuration
        print(f"  - Number of GPUs: {config.num_gpus}")
        print(f"  - GPU Memory Fraction: {config.gpu_memory_fraction}")
        print(f"  - Pipeline Parallel Size: {config.pipeline_parallel_size}")
        print(f"  - Tensor Parallel Size: {config.tensor_parallel_size}")
        print(f"  - Use FP16: {config.use_fp16}")
        print(f"  - Activation Checkpointing: {config.enable_activation_checkpointing}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create XDit+USP config: {e}")
        return False

def test_model_wrapper():
    """Test XDitUSPWanVideoModel wrapper"""
    print("\nTesting XDitUSPWanVideoModel wrapper...")
    
    try:
        from nodes_xdit_usp_loader import XDitUSPWanVideoModel, XDitUSPConfig
        import comfy.model_base
        
        # Create a dummy model config
        model_config = comfy.model_base.ModelConfig(torch.bfloat16)
        
        # Create the wrapper
        model = XDitUSPWanVideoModel(
            model_config,
            model_type=comfy.model_base.ModelType.FLOW,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        print("✓ Created XDitUSPWanVideoModel wrapper")
        
        # Test basic functionality
        model["test_key"] = "test_value"
        assert model["test_key"] == "test_value"
        print("✓ Basic model wrapper functionality works")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create model wrapper: {e}")
        return False

def test_xdit_availability():
    """Test if xdit is available"""
    print("\nTesting xdit availability...")
    
    try:
        import xdit
        print("✓ xdit is available")
        return True
    except ImportError:
        print("✗ xdit is not available - install with: pip install xdit>=0.1.0")
        return False

def test_usp_availability():
    """Test if usp is available"""
    print("\nTesting usp availability...")
    
    try:
        import usp
        print("✓ usp is available")
        return True
    except ImportError:
        print("✗ usp is not available - install with: pip install usp>=0.1.0")
        return False

def test_comfyui_integration():
    """Test ComfyUI integration"""
    print("\nTesting ComfyUI integration...")
    
    try:
        # Test if we can import ComfyUI modules
        import folder_paths
        import comfy.model_management as mm
        import comfy.utils
        import comfy.model_base
        print("✓ ComfyUI modules are available")
        
        # Test device management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        print(f"✓ ComfyUI device management works (main: {device}, offload: {offload_device})")
        
        return True
    except ImportError as e:
        print(f"✗ ComfyUI modules not available: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("XDit+USP Distributed Inference Test Suite")
    print("=" * 60)
    
    tests = [
        ("XDit+USP Imports", test_xdit_usp_imports),
        ("GPU Availability", test_gpu_availability),
        ("XDit+USP Configuration", test_xdit_usp_config),
        ("Model Wrapper", test_model_wrapper),
        ("XDit Availability", test_xdit_availability),
        ("USP Availability", test_usp_availability),
        ("ComfyUI Integration", test_comfyui_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! XDit+USP is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the requirements and dependencies.")
        
        if not any(name == "XDit Availability" and result for name, result in results):
            print("\nTo install xdit:")
            print("  pip install xdit>=0.1.0")
            
        if not any(name == "USP Availability" and result for name, result in results):
            print("\nTo install usp:")
            print("  pip install usp>=0.1.0")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 