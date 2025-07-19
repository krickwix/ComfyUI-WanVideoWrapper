#!/usr/bin/env python3
"""
DistriFusion Model Loader Example
Simple example showing how to use the new streamlined DistriFusion model loader
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from distrifusion.distributed_model_loader import DistriFusionModelLoader, DistriFusionDistributionConfig


def example_basic_usage():
    """Basic usage example with automatic configuration"""
    print("=== DistriFusion Model Loader - Basic Usage ===")
    
    # Create the model loader
    loader = DistriFusionModelLoader()
    
    # Load model with automatic configuration
    model, status = loader.load_model(
        model="your_wanvideo_model.safetensors",  # Replace with your model
        precision="bf16",
        quantization="disabled",
        num_gpus=2,
        split_strategy="auto",  # Automatically choose best strategy
        patch_size=64,
        overlap_ratio=0.125,
        async_comm=True,
        warmup_steps=4,
        memory_optimization="balanced"
    )
    
    print("Model Status:")
    print(status)
    print(f"Model type: {type(model)}")
    

def example_advanced_configuration():
    """Advanced usage with custom configuration"""
    print("\n=== DistriFusion Model Loader - Advanced Configuration ===")
    
    # First create a distribution config
    config_node = DistriFusionDistributionConfig()
    config = config_node.create_config(
        num_gpus=4,
        split_mode="spatial",
        communication_backend="nccl",
        sync_frequency=2,
        patch_overlap=16,
        warmup_steps=6,
        async_communication=True,
        memory_efficient=True,
        debug_mode=False
    )[0]
    
    print("Distribution Config:")
    for key, value in config.items():
        if key != "patch_config":  # Skip the complex object
            print(f"  {key}: {value}")
    
    # Load model with advanced settings
    loader = DistriFusionModelLoader()
    
    model, status = loader.load_model(
        model="your_wanvideo_model.safetensors",
        precision="fp16",
        quantization="fp8_e4m3fn",  # Use quantization for memory efficiency
        num_gpus=4,
        split_strategy="spatial",
        patch_size=128,  # Larger patches for better quality
        overlap_ratio=0.25,  # More overlap for seamless boundaries
        async_comm=True,
        warmup_steps=6,
        memory_optimization="memory",  # Prioritize memory efficiency
        attention_mode="flash_attn_2",  # Use flash attention
        compile_model=True  # Enable compilation for performance
    )
    
    print("Advanced Model Status:")
    print(status)


def example_memory_optimized():
    """Memory-optimized configuration for limited VRAM"""
    print("\n=== DistriFusion Model Loader - Memory Optimized ===")
    
    loader = DistriFusionModelLoader()
    
    model, status = loader.load_model(
        model="your_wanvideo_model.safetensors",
        precision="fp16",
        quantization="fp8_e4m3fn",  # Aggressive quantization
        num_gpus=2,
        split_strategy="temporal",  # Temporal splitting for long videos
        patch_size=32,  # Smaller patches to reduce memory
        overlap_ratio=0.0625,  # Minimal overlap
        async_comm=True,
        warmup_steps=2,  # Fewer warmup steps
        memory_optimization="memory",
        force_rank=0,  # Force to rank 0 for testing
    )
    
    print("Memory-Optimized Model Status:")
    print(status)


def example_performance_optimized():
    """Performance-optimized configuration for maximum speed"""
    print("\n=== DistriFusion Model Loader - Performance Optimized ===")
    
    loader = DistriFusionModelLoader()
    
    model, status = loader.load_model(
        model="your_wanvideo_model.safetensors",
        precision="bf16",
        quantization="disabled",  # No quantization for speed
        num_gpus=4,
        split_strategy="spatial",  # Spatial is fastest
        patch_size=96,
        overlap_ratio=0.125,
        async_comm=True,
        warmup_steps=2,  # Minimal warmup
        memory_optimization="speed",
        attention_mode="flash_attn_3",  # Latest flash attention
        compile_model=True,  # Enable compilation
    )
    
    print("Performance-Optimized Model Status:")
    print(status)


def main():
    """Run all examples"""
    try:
        # Note: These examples will fail without actual models and proper GPU setup
        # They're intended to show the API usage
        
        print("DistriFusion Model Loader Examples")
        print("=" * 50)
        print("Note: These examples show API usage. Replace 'your_wanvideo_model.safetensors'")
        print("      with actual model paths and ensure proper GPU setup.")
        print()
        
        # Run examples (will show API usage even if they fail)
        try:
            example_basic_usage()
        except Exception as e:
            print(f"Basic example failed (expected): {e}")
        
        try:
            example_advanced_configuration()
        except Exception as e:
            print(f"Advanced example failed (expected): {e}")
        
        try:
            example_memory_optimized()
        except Exception as e:
            print(f"Memory-optimized example failed (expected): {e}")
        
        try:
            example_performance_optimized()
        except Exception as e:
            print(f"Performance-optimized example failed (expected): {e}")
        
        print("\n" + "=" * 50)
        print("Examples completed. Check the API calls above for usage patterns.")
        
    except Exception as e:
        print(f"Example error: {e}")


if __name__ == "__main__":
    main() 