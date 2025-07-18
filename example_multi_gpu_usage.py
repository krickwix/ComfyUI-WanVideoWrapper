#!/usr/bin/env python3
"""
Example script demonstrating multi-GPU parallelism with WanVideo models.

This script shows how to:
1. Check available GPUs
2. Determine optimal parallelism configuration
3. Load a model with multi-GPU support
4. Monitor performance and memory usage
"""

import torch
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_gpu_utils import (
        get_gpu_info, 
        print_gpu_status, 
        setup_optimal_parallelism, 
        validate_gpu_setup,
        monitor_memory_usage
    )
except ImportError as e:
    print(f"Error importing multi_gpu_utils: {e}")
    print("Make sure multi_gpu_utils.py is in the same directory")
    sys.exit(1)

def main():
    print("=== WanVideo Multi-GPU Parallelism Example ===\n")
    
    # Step 1: Check CUDA availability
    print("1. Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Multi-GPU parallelism requires CUDA.")
        return
    
    print(f"✅ CUDA is available")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    # Step 2: Get detailed GPU information
    print("\n2. Getting GPU information...")
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("❌ No GPU information available")
        return
    
    print("✅ GPU Information:")
    for gpu_id, info in gpu_info.items():
        print(f"   GPU {gpu_id}: {info['name']}")
        if info['memory_total']:
            print(f"     Memory: {info['memory_total']:.1f} GB")
        if info['temperature']:
            print(f"     Temperature: {info['temperature']}°C")
    
    # Step 3: Print current GPU status
    print("\n3. Current GPU status:")
    print_gpu_status()
    
    # Step 4: Determine optimal configuration for different model sizes
    print("\n4. Optimal configurations for different model sizes:")
    
    model_sizes = [
        ("WanVideo 1.3B", 1.3),
        ("WanVideo 14B", 14.0),
        ("WanVideo 14B (quantized)", 7.0)
    ]
    
    for model_name, size_gb in model_sizes:
        print(f"\n   {model_name} ({size_gb}GB):")
        config = setup_optimal_parallelism("dummy_path.safetensors")  # Size will be overridden
        config['reason'] = f"Model size ({size_gb:.1f}GB) analysis"
        
        # Override the model size for this analysis
        total_memory = sum(gpu_info[gpu_id]['memory_total'] for gpu_id in config['gpu_ids'])
        
        if size_gb <= total_memory * 0.7:
            config['parallelism_type'] = 'data_parallel'
            config['reason'] = f"Model fits in total memory ({total_memory:.1f}GB)"
        elif size_gb > total_memory * 0.7:
            config['parallelism_type'] = 'block_distribution'
            config['reason'] = f"Model exceeds single GPU memory, use block distribution"
        
        print(f"     Recommended: {config['parallelism_type']}")
        print(f"     GPUs: {config['gpu_ids']}")
        print(f"     Reason: {config['reason']}")
    
    # Step 5: Validate GPU setup
    print("\n5. Validating GPU setup...")
    available_gpus = list(gpu_info.keys())
    is_valid, message = validate_gpu_setup(available_gpus)
    
    if is_valid:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    
    # Step 6: Show ComfyUI configuration examples
    print("\n6. ComfyUI Node Configuration Examples:")
    
    print("\n   For WanVideo 14B with 2x RTX 4090 (24GB each):")
    print("   ```json")
    print("   {")
    print('     "model": "wanvideo_14b.safetensors",')
    print('     "base_precision": "bf16",')
    print('     "load_device": "main_device",')
    print('     "parallelism_type": "data_parallel",')
    print('     "gpu_ids": "0,1",')
    print('     "enable_block_distribution": false')
    print("   }")
    print("   ```")
    
    print("\n   For WanVideo 14B with 4x RTX 3080 (10GB each):")
    print("   ```json")
    print("   {")
    print('     "model": "wanvideo_14b.safetensors",')
    print('     "base_precision": "bf16",')
    print('     "load_device": "main_device",')
    print('     "parallelism_type": "block_distribution",')
    print('     "gpu_ids": "0,1,2,3",')
    print('     "enable_block_distribution": true')
    print("   }")
    print("   ```")
    
    # Step 7: Performance tips
    print("\n7. Performance Optimization Tips:")
    print("   • Use bf16 precision for best performance/memory balance")
    print("   • Enable block swapping for large models")
    print("   • Monitor GPU temperatures during long runs")
    print("   • Use appropriate batch sizes for your GPU setup")
    print("   • Consider quantization for memory-constrained setups")
    
    # Step 8: Troubleshooting tips
    print("\n8. Common Issues and Solutions:")
    print("   • CUDA Out of Memory: Reduce batch size or enable block distribution")
    print("   • GPU Communication Errors: Check GPU compatibility and drivers")
    print("   • Performance Issues: Monitor GPU utilization and memory usage")
    print("   • Model Loading Failures: Verify model file integrity and GPU memory")
    
    print("\n=== Example completed successfully! ===")
    print("\nTo use multi-GPU parallelism in ComfyUI:")
    print("1. Add the 'WanVideo Multi-GPU Loader' node to your workflow")
    print("2. Configure the parallelism settings based on the recommendations above")
    print("3. Connect the node to your video generation workflow")
    print("4. Monitor performance and adjust settings as needed")

def test_multi_gpu_loading():
    """
    Test function to demonstrate multi-GPU loading (requires actual model file)
    """
    print("\n=== Testing Multi-GPU Loading ===")
    
    # This would require an actual model file
    model_path = "path/to/your/wanvideo_model.safetensors"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please update the model_path variable with a valid model file")
        return
    
    print(f"✅ Model file found: {model_path}")
    
    # Get optimal configuration
    config = setup_optimal_parallelism(model_path)
    print(f"   Optimal configuration: {config}")
    
    # Validate setup
    is_valid, message = validate_gpu_setup(config['gpu_ids'])
    print(f"   Setup validation: {message}")
    
    print("\n   To load the model with this configuration:")
    print("   1. Use the 'WanVideo Multi-GPU Loader' node in ComfyUI")
    print("   2. Set parallelism_type to:", config['parallelism_type'])
    print("   3. Set gpu_ids to:", ','.join(map(str, config['gpu_ids'])))
    print("   4. Set enable_block_distribution to:", config['enable_block_distribution'])

if __name__ == "__main__":
    try:
        main()
        
        # Uncomment the line below to test with an actual model file
        # test_multi_gpu_loading()
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        import traceback
        traceback.print_exc() 