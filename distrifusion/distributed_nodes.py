"""
ComfyUI Nodes for DistriFusion Distributed Inference
Provides user-friendly nodes for multi-GPU WanVideo inference
"""

import torch
import os
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

try:
    from ..utils import log
except ImportError:
    # Fallback for different import contexts
    import logging
    log = logging.getLogger(__name__)

try:
    from ..nodes_model_loading import WanVideoModelLoader
except ImportError:
    # Fallback for different import contexts
    try:
        from nodes_model_loading import WanVideoModelLoader
    except ImportError:
        WanVideoModelLoader = None
        print("Warning: WanVideoModelLoader not available")

try:
    from ..nodes import WanVideoSampler
except ImportError:
    # Fallback for different import contexts
    try:
        from nodes import WanVideoSampler
    except ImportError:
        WanVideoSampler = None
        print("Warning: WanVideoSampler not available")
from .distrifusion_wrapper import DistriFusionWanModel, create_distrifusion_model
from .patch_manager import PatchConfig, PatchSplitMode
from typing import Dict, Any, Tuple, List, Optional


class DistriFusionWanVideoModelLoader:
    """
    ComfyUI Node for loading WanVideo models with DistriFusion support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), 
                         {"tooltip": "WanVideo model to load with DistriFusion support"}),
                "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
                "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'fp8_e4m3fn_fast_no_ffn'], 
                               {"default": 'disabled', "tooltip": "Optional quantization method"}),
                "load_device": (["main_device", "offload_device"], 
                              {"default": "main_device", "tooltip": "Initial device to load the model to"}),
                
                # DistriFusion specific parameters
                "enable_distrifusion": ("BOOLEAN", {"default": True, "tooltip": "Enable DistriFusion distributed inference"}),
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of GPUs to use for inference"}),
                "split_mode": (["spatial", "temporal", "spatiotemporal"], 
                             {"default": "spatial", "tooltip": "How to split video patches across GPUs"}),
                "patch_overlap": ("INT", {"default": 8, "min": 0, "max": 32, "tooltip": "Overlap size for patch boundaries"}),
                "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 10, "tooltip": "Number of warmup steps with full sync"}),
                
                # Process rank (usually set by launcher)
                "process_rank": ("INT", {"default": 0, "min": 0, "max": 7, "tooltip": "Process rank (set by distributed launcher)"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2", 
                    "flash_attn_3",
                    "sageattn",
                    "flex_attention",
                    "radial_sage_attention",
                ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS",),
                "block_swap_args": ("BLOCKSWAPARGS",),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None}),
                "vace_model": ("VACEPATH", {"default": None}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None}),
            }
        }

    RETURN_TYPES = ("DISTRIFUSION_WANVIDEOMODEL",)
    RETURN_NAMES = ("distrifusion_model",)
    FUNCTION = "load_distrifusion_model"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Load WanVideo model with DistriFusion multi-GPU support"

    def load_distrifusion_model(self, 
                               model, base_precision, quantization, load_device,
                               enable_distrifusion, num_gpus, split_mode, patch_overlap, warmup_steps,
                               process_rank, attention_mode="sdpa", compile_args=None, block_swap_args=None,
                               lora=None, vram_management_args=None, vace_model=None, 
                               fantasytalking_model=None, multitalk_model=None):
        
        # Check if distributed environment is available
        if enable_distrifusion and num_gpus > 1:
            available_gpus = torch.cuda.device_count()
            if available_gpus < num_gpus:
                log.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
                num_gpus = available_gpus
            
            if num_gpus < 2:
                log.warning("DistriFusion requires at least 2 GPUs. Falling back to single GPU mode.")
                enable_distrifusion = False

        # First load the regular WanVideo model
        if WanVideoModelLoader is None:
            raise ImportError(
                "WanVideoModelLoader is not available. Please ensure that the "
                "nodes_model_loading module is properly accessible."
            )
        
        model_loader = WanVideoModelLoader()
        wan_model_patcher = model_loader.loadmodel(
            model=model,
            base_precision=base_precision,
            quantization=quantization,
            load_device=load_device,
            attention_mode=attention_mode,
            compile_args=compile_args,
            block_swap_args=block_swap_args,
            lora=lora,
            vram_management_args=vram_management_args,
            vace_model=vace_model,
            fantasytalking_model=fantasytalking_model,
            multitalk_model=multitalk_model
        )[0]  # Extract the model patcher

        # Get the underlying WanModel
        wan_model = wan_model_patcher.model.diffusion_model

        if enable_distrifusion and num_gpus > 1:
            log.info(f"Creating DistriFusion model with {num_gpus} GPUs")
            
            # Create DistriFusion wrapper
            distrifusion_model = create_distrifusion_model(
                wan_model=wan_model,
                num_devices=num_gpus,
                split_mode=split_mode,
                patch_overlap=patch_overlap,
                world_size=num_gpus,
                rank=process_rank
            )
            
            # Update warmup steps
            distrifusion_model.patch_config.warmup_steps = warmup_steps
            
            # Wrap the patcher to include DistriFusion
            wan_model_patcher.model.diffusion_model = distrifusion_model
            wan_model_patcher.model["distrifusion_enabled"] = True
            wan_model_patcher.model["num_gpus"] = num_gpus
            wan_model_patcher.model["split_mode"] = split_mode
            
            log.info(f"DistriFusion model created successfully on rank {process_rank}/{num_gpus}")
        else:
            # Single GPU mode
            wan_model_patcher.model["distrifusion_enabled"] = False
            log.info("DistriFusion disabled - using single GPU mode")

        return (wan_model_patcher,)


class DistriFusionWanVideoSampler:
    """
    ComfyUI Node for sampling with DistriFusion distributed models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distrifusion_model": ("DISTRIFUSION_WANVIDEOMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["ddim", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast", "dpm_adaptive", "dpmpp_2m_sde", "dpmpp_3m_sde"], 
                               {"default": "ddim"}),
                "scheduler": (["normal", "karras", "exponential", "polyexponential", "sgm_uniform", "simple", "ddim_uniform", "beta"], 
                            {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sync_frequency": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "How often to synchronize between GPUs (every N steps)"}),
                "enable_async_comm": ("BOOLEAN", {"default": True, "tooltip": "Enable asynchronous communication for better performance"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_distrifusion"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Sample using DistriFusion distributed inference"

    def sample_distrifusion(self, distrifusion_model, positive, negative, latent_image, seed, steps, cfg,
                           sampler_name, scheduler, denoise, sync_frequency=1, enable_async_comm=True):
        
        # Check if DistriFusion is enabled
        is_distrifusion_enabled = distrifusion_model.model.get("distrifusion_enabled", False)
        
        if is_distrifusion_enabled:
            log.info("Running DistriFusion distributed sampling")
            
            # Get the DistriFusion model
            diffusion_model = distrifusion_model.model.diffusion_model
            
            if isinstance(diffusion_model, DistriFusionWanModel):
                # Update communication settings
                diffusion_model.patch_config.async_boundary_update = enable_async_comm
                
                log.info(f"DistriFusion settings: GPUs={distrifusion_model.model.get('num_gpus', 2)}, "
                        f"Mode={distrifusion_model.model.get('split_mode', 'spatial')}, "
                        f"Async={enable_async_comm}")
        
        # Use the regular WanVideo sampler but with potentially distributed model
        sampler = WanVideoSampler()
        
        # Create a custom callback to update DistriFusion step
        original_callback = getattr(sampler, 'callback', None)
        
        def distrifusion_callback(step, x0, x, total_steps):
            # Update DistriFusion step tracking
            if is_distrifusion_enabled and isinstance(distrifusion_model.model.diffusion_model, DistriFusionWanModel):
                distrifusion_model.model.diffusion_model.update_step(step)
                
                # Periodic synchronization
                if step % sync_frequency == 0:
                    distrifusion_model.model.diffusion_model.synchronize()
            
            # Call original callback if it exists
            if original_callback:
                return original_callback(step, x0, x, total_steps)
        
        # Set the callback
        sampler.callback = distrifusion_callback
        
        try:
            # Run sampling
            result = sampler.sample(
                model=distrifusion_model,
                conditioning=positive,
                neg_conditioning=negative,
                latent_image=latent_image,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            # Final synchronization for DistriFusion
            if is_distrifusion_enabled and isinstance(distrifusion_model.model.diffusion_model, DistriFusionWanModel):
                distrifusion_model.model.diffusion_model.synchronize()
                log.info("DistriFusion sampling completed successfully")
            
            return result
            
        except Exception as e:
            log.error(f"DistriFusion sampling failed: {e}")
            # Cleanup on error
            if is_distrifusion_enabled and isinstance(distrifusion_model.model.diffusion_model, DistriFusionWanModel):
                distrifusion_model.model.diffusion_model.cleanup()
            raise


class DistriFusionSetup:
    """
    Utility node for setting up DistriFusion distributed environment
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Total number of processes"}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 7, "tooltip": "Current process rank"}),
                "master_addr": ("STRING", {"default": "localhost", "tooltip": "Master node address"}),
                "master_port": ("STRING", {"default": "12355", "tooltip": "Master node port"}),
                "backend": (["nccl", "gloo"], {"default": "nccl", "tooltip": "Communication backend"}),
            }
        }

    RETURN_TYPES = ("DISTRIFUSION_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "setup_distributed"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Setup distributed environment for DistriFusion"

    def setup_distributed(self, world_size, rank, master_addr, master_port, backend):
        import torch.distributed as dist
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        # Initialize distributed environment if not already initialized
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size
                )
                log.info(f"Initialized distributed group: rank {rank}/{world_size}")
            except Exception as e:
                log.error(f"Failed to initialize distributed group: {e}")
                raise
        
        config = {
            "world_size": world_size,
            "rank": rank,
            "master_addr": master_addr,
            "master_port": master_port,
            "backend": backend,
            "initialized": True
        }
        
        return (config,)


class DistriFusionStatus:
    """
    Node to check DistriFusion status and performance metrics
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distrifusion_model": ("DISTRIFUSION_WANVIDEOMODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get_status"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Get DistriFusion status and performance information"

    def get_status(self, distrifusion_model):
        status_info = []
        
        # Check if DistriFusion is enabled
        is_enabled = distrifusion_model.model.get("distrifusion_enabled", False)
        status_info.append(f"DistriFusion Enabled: {is_enabled}")
        
        if is_enabled:
            num_gpus = distrifusion_model.model.get("num_gpus", 1)
            split_mode = distrifusion_model.model.get("split_mode", "spatial")
            
            status_info.append(f"Number of GPUs: {num_gpus}")
            status_info.append(f"Split Mode: {split_mode}")
            
            # Get model info
            diffusion_model = distrifusion_model.model.diffusion_model
            if isinstance(diffusion_model, DistriFusionWanModel):
                status_info.append(f"Current Step: {diffusion_model.current_step}")
                status_info.append(f"World Size: {diffusion_model.world_size}")
                status_info.append(f"Rank: {diffusion_model.rank}")
                status_info.append(f"Device: {diffusion_model.device}")
                
                # Patch config info
                config = diffusion_model.patch_config
                status_info.append(f"Patch Overlap: {config.patch_overlap}")
                status_info.append(f"Warmup Steps: {config.warmup_steps}")
                status_info.append(f"Async Updates: {config.async_boundary_update}")
        
        # GPU memory info
        status_info.append("\n=== GPU Memory Status ===")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            status_info.append(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
        return ("\n".join(status_info),)


# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DistriFusionWanVideoModelLoader": DistriFusionWanVideoModelLoader,
    "DistriFusionWanVideoSampler": DistriFusionWanVideoSampler,
    "DistriFusionSetup": DistriFusionSetup,
    "DistriFusionStatus": DistriFusionStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DistriFusionWanVideoModelLoader": "DistriFusion WanVideo Model Loader",
    "DistriFusionWanVideoSampler": "DistriFusion WanVideo Sampler", 
    "DistriFusionSetup": "DistriFusion Setup",
    "DistriFusionStatus": "DistriFusion Status",
} 