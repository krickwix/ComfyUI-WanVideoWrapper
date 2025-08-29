#!/usr/bin/env python3
"""
Test script to verify that the distributed inference options are working correctly
"""

import sys
import os

# Add the current directory to the path so we can import the nodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_distributed_options():
    """Test that the distributed options are properly defined in the model loader"""
    try:
        from nodes_model_loading import WanVideoModelLoader
        
        # Get the input types
        input_types = WanVideoModelLoader.INPUT_TYPES()
        
        # Check if distributed options are present
        optional_inputs = input_types.get("optional", {})
        
        print("=== Testing Distributed Options in LoadWanVideoModel ===")
        print(f"Optional inputs found: {list(optional_inputs.keys())}")
        
        # Check for specific distributed options
        distributed_options = [
            "enable_distributed",
            "gpu_count", 
            "use_ulysses",
            "use_fsdp",
            "master_port"
        ]
        
        missing_options = []
        for option in distributed_options:
            if option in optional_inputs:
                print(f"✓ {option}: {optional_inputs[option]}")
            else:
                print(f"✗ {option}: MISSING")
                missing_options.append(option)
        
        if missing_options:
            print(f"\n❌ Missing distributed options: {missing_options}")
            return False
        else:
            print("\n✅ All distributed options are present!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing distributed options: {e}")
        return False

def test_distributed_nodes():
    """Test that the distributed inference nodes are properly defined"""
    try:
        from nodes_distributed import WanVideoDistributedInference, WanVideoDistributedConfig
        
        print("\n=== Testing Distributed Inference Nodes ===")
        
        # Test WanVideoDistributedInference
        inference_inputs = WanVideoDistributedInference.INPUT_TYPES()
        print(f"✓ WanVideoDistributedInference inputs: {list(inference_inputs.get('required', {}).keys())}")
        
        # Test WanVideoDistributedConfig
        config_inputs = WanVideoDistributedConfig.INPUT_TYPES()
        print(f"✓ WanVideoDistributedConfig inputs: {list(config_inputs.get('required', {}).keys())}")
        
        print("✅ Distributed inference nodes are properly defined!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing distributed nodes: {e}")
        return False

def test_node_registration():
    """Test that the nodes are properly registered"""
    try:
        from __init__ import NODE_CLASS_MAPPINGS
        
        print("\n=== Testing Node Registration ===")
        
        # Check if distributed nodes are registered
        distributed_nodes = [
            "WanVideoDistributedInference",
            "WanVideoDistributedConfig"
        ]
        
        missing_nodes = []
        for node in distributed_nodes:
            if node in NODE_CLASS_MAPPINGS:
                print(f"✓ {node}: Registered")
            else:
                print(f"✗ {node}: NOT REGISTERED")
                missing_nodes.append(node)
        
        if missing_nodes:
            print(f"\n❌ Missing registered nodes: {missing_nodes}")
            return False
        else:
            print("\n✅ All distributed nodes are registered!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing node registration: {e}")
        return False

if __name__ == "__main__":
    print("Testing ComfyUI-WanVideoWrapper Distributed Inference Setup...\n")
    
    # Run all tests
    test1 = test_distributed_options()
    test2 = test_distributed_nodes()
    test3 = test_node_registration()
    
    print("\n" + "="*50)
    if all([test1, test2, test3]):
        print("🎉 ALL TESTS PASSED! Distributed inference should work correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for issues.")
    print("="*50)
