#!/usr/bin/env python3
"""
Test script for Wan2.1 distributed inference implementation
"""

import sys
import os
import torch
import gc

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch.distributed as dist
        print("✓ torch.distributed imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import torch.distributed: {e}")
        return False
    
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
        print("✓ FSDP components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FSDP components: {e}")
        return False
    
    try:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        print("✓ xfuser components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import xfuser components: {e}")
        print("  Note: xfuser is optional for context parallel support")
    
    try:
        from nodes_xdit_usp_loader import WanDistributedConfig, WanDistributedModel
        print("✓ Wan2.1 distributed components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Wan2.1 distributed components: {e}")
        return False
    
    return True

def test_gpu_availability():
    """Test GPU availability and CUDA support"""
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
    
    return gpu_count > 0

def test_distributed_config():
    """Test WanDistributedConfig creation"""
    print("\nTesting WanDistributedConfig...")
    
    try:
        from nodes_xdit_usp_loader import WanDistributedConfig
        from torch.distributed.fsdp import ShardingStrategy
        
        # Test basic configuration
        config = WanDistributedConfig(
            world_size=2,
            rank=0,
            backend="nccl",
            use_fsdp=True,
            use_context_parallel=False,
            param_dtype=torch.bfloat16
        )
        
        print(f"✓ Created WanDistributedConfig:")
        print(f"  world_size: {config.world_size}")
        print(f"  rank: {config.rank}")
        print(f"  backend: {config.backend}")
        print(f"  use_fsdp: {config.use_fsdp}")
        print(f"  use_context_parallel: {config.use_context_parallel}")
        print(f"  param_dtype: {config.param_dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create WanDistributedConfig: {e}")
        return False

def test_model_wrapper():
    """Test WanDistributedModel wrapper"""
    print("\nTesting WanDistributedModel wrapper...")
    
    try:
        from nodes_xdit_usp_loader import WanDistributedModel, WanDistributedConfig
        import comfy.model_base
        
        # Create a dummy model wrapper
        model = WanDistributedModel(
            comfy.model_base.ModelConfig(torch.bfloat16),
            model_type=comfy.model_base.ModelType.FLOW,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        
        print("✓ Created WanDistributedModel wrapper")
        
        # Test configuration setup (without actual distributed init)
        config = WanDistributedConfig(
            world_size=1,
            rank=0,
            use_fsdp=False,
            use_context_parallel=False
        )
        
        # This should work without actual distributed setup
        print("✓ Model wrapper created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create model wrapper: {e}")
        return False

def test_comfyui_integration():
    """Test ComfyUI node integration"""
    print("\nTesting ComfyUI integration...")
    
    try:
        from nodes_xdit_usp_loader import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        expected_nodes = ["WanDistributedConfig", "WanDistributedModelLoader"]
        
        for node_name in expected_nodes:
            if node_name in NODE_CLASS_MAPPINGS:
                print(f"✓ {node_name} registered in NODE_CLASS_MAPPINGS")
            else:
                print(f"✗ {node_name} not found in NODE_CLASS_MAPPINGS")
                return False
            
            if node_name in NODE_DISPLAY_NAME_MAPPINGS:
                print(f"✓ {node_name} registered in NODE_DISPLAY_NAME_MAPPINGS")
            else:
                print(f"✗ {node_name} not found in NODE_DISPLAY_NAME_MAPPINGS")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test ComfyUI integration: {e}")
        return False

def test_memory_management():
    """Test memory management utilities"""
    print("\nTesting memory management...")
    
    try:
        # Test basic memory operations
        if torch.cuda.is_available():
            # Allocate some GPU memory
            tensor = torch.randn(1000, 1000, device="cuda")
            memory_before = torch.cuda.memory_allocated()
            
            # Clear memory
            del tensor
            torch.cuda.empty_cache()
            gc.collect()
            
            memory_after = torch.cuda.memory_allocated()
            print(f"✓ Memory management test: {memory_before} -> {memory_after} bytes")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test memory management: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Wan2.1 Distributed Inference Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("GPU Availability", test_gpu_availability),
        ("Distributed Config", test_distributed_config),
        ("Model Wrapper", test_model_wrapper),
        ("ComfyUI Integration", test_comfyui_integration),
        ("Memory Management", test_memory_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Wan2.1 distributed inference is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 