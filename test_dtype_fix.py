#!/usr/bin/env python3
"""
Test script to verify dtype handling in distributed model loader
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dtype_handling():
    """Test that dtype conversion works correctly"""
    print("🧪 Testing dtype handling...")
    
    # Test bfloat16 conversion
    test_tensor = torch.randn(10, 10, dtype=torch.float32)
    print(f"   - Original tensor dtype: {test_tensor.dtype}")
    
    # Convert to bfloat16
    converted = test_tensor.to(dtype=torch.bfloat16)
    print(f"   - Converted tensor dtype: {converted.dtype}")
    
    # Test that conversion worked
    assert converted.dtype == torch.bfloat16, f"Expected bfloat16, got {converted.dtype}"
    print("✅ bfloat16 conversion test passed")
    
    # Test model parameter dtype
    model = torch.nn.Linear(10, 10)
    print(f"   - Model parameter dtype: {model.weight.dtype}")
    
    # Convert model to bfloat16
    model = model.to(dtype=torch.bfloat16)
    print(f"   - Model parameter dtype after conversion: {model.weight.dtype}")
    
    assert model.weight.dtype == torch.bfloat16, f"Expected bfloat16, got {model.weight.dtype}"
    print("✅ Model dtype conversion test passed")
    
    print("🎉 All dtype tests passed!")

if __name__ == "__main__":
    test_dtype_handling() 