#!/usr/bin/env python3
"""
DistriFusion Example Script
Demonstrates how to use DistriFusion for distributed WanVideo inference
"""

import torch
import torch.distributed as dist
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from distrifusion import (
    DistriFusionWanVideoModelLoader,
    DistriFusionWanVideoSampler,
    DistriFusionSetup,
    create_distrifusion_model
)
from nodes_model_loading import WanVideoModelLoader


def setup_distributed(rank, world_size, master_port=12355):
    """Setup distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    print(f"Process {rank}/{world_size} initialized on GPU {rank}")


def create_sample_input(batch_size=1, channels=16, frames=21, height=128, width=128, device='cuda'):
    """Create sample latent input for testing"""
    latent = torch.randn(batch_size, channels, frames, height, width, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create dummy context (text embeddings)
    context = [torch.randn(batch_size, 512, 4096, device=device)]
    
    return latent, timestep, context


def main():
    parser = argparse.ArgumentParser(description="DistriFusion Example")
    parser.add_argument("--model", type=str, required=True, help="Path to WanVideo model")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--rank", type=int, default=0, help="Process rank")
    parser.add_argument("--split-mode", type=str, default="spatial", 
                       choices=["spatial", "temporal", "spatiotemporal"])
    parser.add_argument("--patch-overlap", type=int, default=8, help="Patch overlap size")
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--frames", type=int, default=21, help="Number of frames")
    
    args = parser.parse_args()
    
    # Setup distributed environment
    if args.gpus > 1:
        setup_distributed(args.rank, args.gpus)
    
    try:
        # Method 1: Using DistriFusion nodes (recommended)
        print("=== Method 1: Using DistriFusion Nodes ===")
        example_with_nodes(args)
        
        # Method 2: Manual setup (advanced)
        print("\n=== Method 2: Manual DistriFusion Setup ===")
        example_manual_setup(args)
        
    except Exception as e:
        print(f"Error in example: {e}")
        raise
    finally:
        if args.gpus > 1 and dist.is_initialized():
            dist.destroy_process_group()


def example_with_nodes(args):
    """Example using DistriFusion ComfyUI nodes"""
    
    # Load model with DistriFusion
    model_loader = DistriFusionWanVideoModelLoader()
    
    distrifusion_model = model_loader.load_distrifusion_model(
        model=args.model,
        base_precision="bf16",
        quantization="disabled",
        load_device="main_device",
        enable_distrifusion=args.gpus > 1,
        num_gpus=args.gpus,
        split_mode=args.split_mode,
        patch_overlap=args.patch_overlap,
        warmup_steps=4,
        process_rank=args.rank,
        attention_mode="sdpa"
    )[0]
    
    print(f"Model loaded with DistriFusion: {distrifusion_model.model.get('distrifusion_enabled', False)}")
    
    # Create sample input
    device = f"cuda:{args.rank}" if args.gpus > 1 else "cuda"
    latent_input, timestep, context = create_sample_input(
        frames=args.frames,
        height=args.height // 8,  # Latent space is 8x downscaled
        width=args.width // 8,
        device=device
    )
    
    # Create dummy conditioning
    positive_conditioning = [context[0], {}]  # (context, pooled)
    negative_conditioning = [torch.zeros_like(context[0]), {}]
    
    # Create latent dict for ComfyUI format
    latent_dict = {
        "samples": latent_input
    }
    
    # Sample with DistriFusion
    if args.gpus > 1:
        print(f"Running DistriFusion inference on GPU {args.rank}/{args.gpus}")
    else:
        print("Running single GPU inference")
    
    sampler = DistriFusionWanVideoSampler()
    
    result = sampler.sample_distrifusion(
        distrifusion_model=distrifusion_model,
        positive=positive_conditioning,
        negative=negative_conditioning,
        latent_image=latent_dict,
        seed=42,
        steps=args.steps,
        cfg=7.5,
        sampler_name="ddim",
        scheduler="normal",
        denoise=1.0,
        sync_frequency=2,
        enable_async_comm=True
    )
    
    output_latents = result[0]["samples"]
    print(f"Generated video latents shape: {output_latents.shape}")
    
    # Only save on main process
    if args.rank == 0:
        print("Inference completed successfully!")
        print(f"Output shape: {output_latents.shape}")
        
        # Save latents for inspection
        torch.save(output_latents, "distrifusion_output.pt")
        print("Saved output latents to distrifusion_output.pt")


def example_manual_setup(args):
    """Example with manual DistriFusion setup (advanced)"""
    
    # Load regular WanVideo model first
    model_loader = WanVideoModelLoader()
    wan_model_patcher = model_loader.loadmodel(
        model=args.model,
        base_precision="bf16",
        quantization="disabled",
        load_device="main_device",
        attention_mode="sdpa"
    )[0]
    
    wan_model = wan_model_patcher.model.diffusion_model
    
    if args.gpus > 1:
        # Create DistriFusion wrapper
        distrifusion_model = create_distrifusion_model(
            wan_model=wan_model,
            num_devices=args.gpus,
            split_mode=args.split_mode,
            patch_overlap=args.patch_overlap,
            world_size=args.gpus,
            rank=args.rank
        )
        
        print(f"Created DistriFusion model on rank {args.rank}")
        
        # Manual inference loop
        device = f"cuda:{args.rank}"
        latent_input, timestep, context = create_sample_input(
            frames=args.frames,
            height=args.height // 8,
            width=args.width // 8,
            device=device
        )
        
        print("Running manual DistriFusion inference...")
        
        # Simulate denoising steps
        for step in range(args.steps):
            distrifusion_model.update_step(step)
            
            # Forward pass
            output, pred_id = distrifusion_model(
                x=latent_input,
                t=timestep,
                context=context
            )
            
            # Synchronize every few steps
            if step % 2 == 0:
                distrifusion_model.synchronize()
            
            print(f"Step {step+1}/{args.steps} completed on rank {args.rank}")
        
        # Final synchronization
        distrifusion_model.synchronize()
        
        if args.rank == 0:
            print("Manual DistriFusion inference completed!")
            output_shape = output.shape if isinstance(output, torch.Tensor) else output[0].shape
            print(f"Output shape: {output_shape}")
        
        # Cleanup
        distrifusion_model.cleanup()
        
    else:
        print("Single GPU mode - skipping manual DistriFusion example")


if __name__ == "__main__":
    main() 