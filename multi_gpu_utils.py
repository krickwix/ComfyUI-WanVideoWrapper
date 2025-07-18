"""
Multi-GPU Parallelism Utilities for WanVideo

This module provides utility functions for setting up and monitoring
multi-GPU parallelism when loading WanVideo models.
"""

import torch
import logging
from typing import List, Dict, Optional, Tuple
import psutil
import GPUtil

log = logging.getLogger(__name__)

def get_gpu_info() -> Dict[int, Dict]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        Dictionary mapping GPU ID to GPU information
    """
    gpu_info = {}
    
    if not torch.cuda.is_available():
        log.warning("CUDA is not available")
        return gpu_info
    
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info[gpu.id] = {
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'temperature': gpu.temperature,
                'load': gpu.load * 100 if gpu.load else 0,
                'uuid': gpu.uuid
            }
    except Exception as e:
        log.warning(f"Could not get detailed GPU info: {e}")
        # Fallback to basic torch info
        for i in range(torch.cuda.device_count()):
            gpu_info[i] = {
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory // (1024**3),  # GB
                'memory_used': None,
                'memory_free': None,
                'temperature': None,
                'load': None,
                'uuid': None
            }
    
    return gpu_info

def print_gpu_status():
    """
    Print current GPU status and memory usage.
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("No GPUs available")
        return
    
    print("\n=== GPU Status ===")
    for gpu_id, info in gpu_info.items():
        print(f"GPU {gpu_id}: {info['name']}")
        if info['memory_total']:
            if info['memory_used'] is not None:
                print(f"  Memory: {info['memory_used']}/{info['memory_total']} MB "
                      f"({info['memory_used']/info['memory_total']*100:.1f}% used)")
            else:
                print(f"  Memory: {info['memory_total']} GB total")
        
        if info['temperature']:
            print(f"  Temperature: {info['temperature']}°C")
        if info['load'] is not None:
            print(f"  Load: {info['load']:.1f}%")
        print()

def get_optimal_gpu_config(model_size_gb: float = 14.0) -> Tuple[List[int], str]:
    """
    Automatically determine optimal GPU configuration for a given model size.
    
    Args:
        model_size_gb: Size of the model in GB (default: 14GB for WanVideo 14B)
    
    Returns:
        Tuple of (gpu_ids, parallelism_type)
    """
    gpu_info = get_gpu_info()
    available_gpus = list(gpu_info.keys())
    
    if not available_gpus:
        return [], "none"
    
    # Calculate total available memory
    total_memory = sum(gpu_info[gpu_id]['memory_total'] for gpu_id in available_gpus)
    
    # Simple heuristic for parallelism type
    if len(available_gpus) == 1:
        return available_gpus, "none"
    elif model_size_gb <= total_memory * 0.8:  # Model fits in total memory
        return available_gpus, "data_parallel"
    else:
        return available_gpus, "block_distribution"

def monitor_memory_usage(model, interval: float = 5.0, duration: float = 60.0):
    """
    Monitor memory usage during model inference.
    
    Args:
        model: The model to monitor
        interval: Monitoring interval in seconds
        duration: Total monitoring duration in seconds
    """
    import time
    import threading
    
    def monitor_loop():
        start_time = time.time()
        while time.time() - start_time < duration:
            print(f"\n=== Memory Usage at {time.time() - start_time:.1f}s ===")
            print_gpu_status()
            
            # Print model device info if available
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                transformer = model.model.diffusion_model
                if hasattr(transformer, 'blocks'):
                    print(f"Model blocks on devices:")
                    for i, block in enumerate(transformer.blocks):
                        print(f"  Block {i}: {next(block.parameters()).device}")
            
            time.sleep(interval)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread

def benchmark_multi_gpu_performance(model, input_data, num_runs: int = 5):
    """
    Benchmark performance across different GPU configurations.
    
    Args:
        model: The model to benchmark
        input_data: Sample input data
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Single GPU baseline
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(*input_data)
        end_time.record()
        
        torch.cuda.synchronize()
        single_gpu_time = start_time.elapsed_time(end_time) / num_runs
        results['single_gpu'] = single_gpu_time
    
    # Multi-GPU if available
    gpu_info = get_gpu_info()
    if len(gpu_info) > 1:
        # Test different parallelism types
        for parallelism_type in ['data_parallel', 'block_distribution']:
            try:
                # This would require reconfiguring the model
                # For now, just log the attempt
                log.info(f"Benchmarking {parallelism_type} not implemented yet")
            except Exception as e:
                log.warning(f"Failed to benchmark {parallelism_type}: {e}")
    
    return results

def setup_optimal_parallelism(model_path: str, available_vram: float = None) -> Dict:
    """
    Setup optimal parallelism configuration based on model and hardware.
    
    Args:
        model_path: Path to the model file
        available_vram: Available VRAM in GB (if None, will be detected)
    
    Returns:
        Dictionary with recommended configuration
    """
    import os
    
    # Estimate model size
    if os.path.exists(model_path):
        model_size_gb = os.path.getsize(model_path) / (1024**3)
    else:
        model_size_gb = 14.0  # Default for WanVideo 14B
    
    # Get GPU info
    gpu_info = get_gpu_info()
    available_gpus = list(gpu_info.keys())
    
    if not available_gpus:
        return {
            'parallelism_type': 'none',
            'gpu_ids': [],
            'enable_block_distribution': False,
            'reason': 'No CUDA devices available'
        }
    
    # Calculate total available memory
    total_memory = sum(gpu_info[gpu_id]['memory_total'] for gpu_id in available_gpus)
    
    # Determine optimal configuration
    if len(available_gpus) == 1:
        config = {
            'parallelism_type': 'none',
            'gpu_ids': available_gpus,
            'enable_block_distribution': False,
            'reason': 'Single GPU available'
        }
    elif model_size_gb <= total_memory * 0.7:  # Model fits comfortably
        config = {
            'parallelism_type': 'data_parallel',
            'gpu_ids': available_gpus,
            'enable_block_distribution': False,
            'reason': f'Model size ({model_size_gb:.1f}GB) fits in total memory ({total_memory:.1f}GB)'
        }
    else:
        config = {
            'parallelism_type': 'block_distribution',
            'gpu_ids': available_gpus,
            'enable_block_distribution': True,
            'reason': f'Model size ({model_size_gb:.1f}GB) exceeds total memory ({total_memory:.1f}GB)'
        }
    
    return config

def validate_gpu_setup(gpu_ids: List[int]) -> Tuple[bool, str]:
    """
    Validate that the specified GPU setup is valid.
    
    Args:
        gpu_ids: List of GPU IDs to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    
    available_gpus = list(range(torch.cuda.device_count()))
    
    if not gpu_ids:
        return False, "No GPU IDs specified"
    
    for gpu_id in gpu_ids:
        if gpu_id not in available_gpus:
            return False, f"GPU {gpu_id} is not available. Available GPUs: {available_gpus}"
    
    # Check if GPUs have enough memory
    gpu_info = get_gpu_info()
    for gpu_id in gpu_ids:
        if gpu_id in gpu_info:
            memory_gb = gpu_info[gpu_id]['memory_total']
            if memory_gb < 8:  # Minimum 8GB recommended
                return False, f"GPU {gpu_id} has only {memory_gb:.1f}GB memory. Minimum 8GB recommended."
    
    return True, "GPU setup is valid"

# Example usage functions
def example_multi_gpu_loading():
    """
    Example of how to use multi-GPU loading in ComfyUI.
    """
    print("=== Multi-GPU WanVideo Loading Example ===")
    
    # 1. Check available GPUs
    print("1. Checking available GPUs:")
    print_gpu_status()
    
    # 2. Get optimal configuration
    print("\n2. Getting optimal configuration:")
    config = setup_optimal_parallelism("path/to/wanvideo_model.safetensors")
    print(f"Recommended config: {config}")
    
    # 3. Validate setup
    print("\n3. Validating GPU setup:")
    is_valid, message = validate_gpu_setup(config['gpu_ids'])
    print(f"Setup valid: {is_valid}")
    print(f"Message: {message}")
    
    print("\n4. ComfyUI Node Configuration:")
    print("Use the 'WanVideo Multi-GPU Loader' node with these settings:")
    print(f"  - parallelism_type: {config['parallelism_type']}")
    print(f"  - gpu_ids: {','.join(map(str, config['gpu_ids']))}")
    print(f"  - enable_block_distribution: {config['enable_block_distribution']}")

if __name__ == "__main__":
    example_multi_gpu_loading() 