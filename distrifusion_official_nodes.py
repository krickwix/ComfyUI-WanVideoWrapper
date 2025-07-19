#!/usr/bin/env python3
"""
Official DistriFusion ComfyUI Nodes
Uses the MIT-HAN-Lab DistriFusion implementation
"""

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

try:
    from utils import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)

try:
    from nodes_model_loading import WanVideoModelLoader
except ImportError:
    try:
        from ..nodes_model_loading import WanVideoModelLoader
    except ImportError:
        WanVideoModelLoader = None
        print("Warning: WanVideoModelLoader not available")

try:
    from nodes import WanVideoSampler
except ImportError:
    try:
        from ..nodes import WanVideoSampler
    except ImportError:
        WanVideoSampler = None
        print("Warning: WanVideoSampler not available")

try:
    from distrifusion_official_wrapper import (
        create_official_distrifusion_model,
        get_official_distrifusion_config,
        check_official_distrifusion_availability,
        DISTRIFUSER_AVAILABLE
    )
    OFFICIAL_DISTRIFUSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Official DistriFusion wrapper not available: {e}")
    OFFICIAL_DISTRIFUSION_AVAILABLE = False


class OfficialDistriFusionModelLoader:
    """
    ComfyUI Node for loading WanVideo models with official DistriFusion
    Uses the MIT-HAN-Lab implementation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), 
                         {"tooltip": "WanVideo model to load with official DistriFusion"}),
                
                # Core model settings
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Model precision"}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e5m2"], {"default": "disabled"}),
                
                # Official DistriFusion settings
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of GPUs for distribution"}),
                "patch_size": ("INT", {"default": 64, "min": 32, "max": 128, "step": 8, "tooltip": "Patch size for splitting"}),
                "patch_overlap": ("INT", {"default": 8, "min": 0, "max": 32, "tooltip": "Overlap size for boundaries"}),
                "split_mode": (["spatial", "temporal", "spatiotemporal"], 
                             {"default": "spatial", "tooltip": "How to split patches across GPUs"}),
                
                # Performance settings
                "world_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Total number of processes"}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 7, "tooltip": "Current process rank"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa", "flash_attn_2", "flash_attn_3", "sageattn", "flex_attention"
                ], {"default": "sdpa"}),
                "lora": ("WANVIDLORA", {"default": None}),
                "compile_model": ("BOOLEAN", {"default": False, "tooltip": "Compile model for better performance"}),
            }
        }

    RETURN_TYPES = ("OFFICIAL_DISTRIFUSION_MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load_model"
    CATEGORY = "WanVideoWrapper/Official DistriFusion"
    DESCRIPTION = "Load WanVideo model with official MIT-HAN-Lab DistriFusion"
    
    def load_model(self, model, precision, quantization, num_gpus, patch_size, patch_overlap, 
                   split_mode, world_size, rank, attention_mode="sdpa", lora=None, compile_model=False):
        
        status_messages = []
        
        # Check if official DistriFusion is available
        if not OFFICIAL_DISTRIFUSION_AVAILABLE:
            error_msg = "Official DistriFusion not available. Install with: pip install git+https://github.com/mit-han-lab/distrifuser.git"
            status_messages.append(f"❌ {error_msg}")
            return (None, "\n".join(status_messages))
        
        # Check availability
        available, msg = check_official_distrifusion_availability()
        if not available:
            status_messages.append(f"❌ {msg}")
            return (None, "\n".join(status_messages))
        
        status_messages.append("✅ Official DistriFusion available")
        
        # Check GPU availability
        if num_gpus > 1:
            if not torch.cuda.is_available():
                status_messages.append("⚠️ CUDA not available, falling back to single GPU")
                num_gpus = 1
            elif torch.cuda.device_count() < num_gpus:
                available_gpus = torch.cuda.device_count()
                status_messages.append(f"⚠️ Only {available_gpus} GPUs available, using {available_gpus}")
                num_gpus = available_gpus
        
        # Load base WanVideo model
        if WanVideoModelLoader is None:
            error_msg = "WanVideoModelLoader not available"
            status_messages.append(f"❌ {error_msg}")
            return (None, "\n".join(status_messages))
        
        try:
            loader = WanVideoModelLoader()
            wan_model_patcher = loader.loadmodel(
                model=model,
                base_precision=precision,
                quantization=quantization,
                load_device="main_device",
                attention_mode=attention_mode,
                lora=lora
            )[0]  # Extract the model patcher
            
            status_messages.append("✅ Base WanVideo model loaded")
            
        except Exception as e:
            error_msg = f"Failed to load base model: {e}"
            status_messages.append(f"❌ {error_msg}")
            return (None, "\n".join(status_messages))
        
        # Create official DistriFusion model
        try:
            distrifusion_model = create_official_distrifusion_model(
                wan_model=wan_model_patcher,
                num_devices=num_gpus,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                split_mode=split_mode,
                world_size=world_size,
                rank=rank
            )
            
            status_messages.append(f"✅ Official DistriFusion model created")
            status_messages.append(f"   - GPUs: {num_gpus}")
            status_messages.append(f"   - Patch size: {patch_size}")
            status_messages.append(f"   - Split mode: {split_mode}")
            
            # Compile if requested
            if compile_model:
                try:
                    distrifusion_model = torch.compile(distrifusion_model)
                    status_messages.append("✅ Model compiled for better performance")
                except Exception as e:
                    status_messages.append(f"⚠️ Model compilation failed: {e}")
            
            return (distrifusion_model, "\n".join(status_messages))
            
        except Exception as e:
            error_msg = f"Failed to create DistriFusion model: {e}"
            status_messages.append(f"❌ {error_msg}")
            return (None, "\n".join(status_messages))


class OfficialDistriFusionSampler:
    """
    ComfyUI Node for sampling with official DistriFusion models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_distrifusion_model": ("OFFICIAL_DISTRIFUSION_MODEL",),
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
                "sync_frequency": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "How often to synchronize between GPUs"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "WanVideoWrapper/Official DistriFusion"
    DESCRIPTION = "Sample using official DistriFusion distributed inference"
    
    def sample(self, official_distrifusion_model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, sync_frequency=1):
        
        if official_distrifusion_model is None:
            raise ValueError("Official DistriFusion model is None")
        
        # Use the WanVideoSampler if available, otherwise implement basic sampling
        if WanVideoSampler is not None:
            # Create a temporary sampler instance
            sampler = WanVideoSampler()
            
            # Call the sampling function with the DistriFusion model
            return sampler.sample(
                model=official_distrifusion_model,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
        else:
            # Basic sampling implementation
            torch.manual_seed(seed)
            
            # Get the latent
            latent = latent_image["samples"]
            
            # Simple sampling loop (this is a basic implementation)
            for step in range(steps):
                # Update DistriFusion step
                official_distrifusion_model.update_step(step)
                
                # Synchronize if needed
                if step % sync_frequency == 0:
                    official_distrifusion_model.synchronize()
                
                # Basic sampling step (this would need proper implementation)
                # For now, just return the input latent
                pass
            
            return ({"samples": latent},)


class OfficialDistriFusionStatus:
    """
    Node to check official DistriFusion status
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_distrifusion_model": ("OFFICIAL_DISTRIFUSION_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get_status"
    CATEGORY = "WanVideoWrapper/Official DistriFusion"
    DESCRIPTION = "Get official DistriFusion status and information"
    
    def get_status(self, official_distrifusion_model):
        if official_distrifusion_model is None:
            return "❌ No official DistriFusion model provided"
        
        try:
            # Get model information
            info = []
            info.append("🎯 Official DistriFusion Model Status")
            info.append("=" * 40)
            
            # Check availability
            available, msg = check_official_distrifusion_availability()
            if available:
                info.append("✅ Official DistriFusion: Available")
            else:
                info.append(f"❌ Official DistriFusion: {msg}")
            
            # Model configuration
            if hasattr(official_distrifusion_model, 'num_devices'):
                info.append(f"📊 GPUs: {official_distrifusion_model.num_devices}")
            if hasattr(official_distrifusion_model, 'patch_size'):
                info.append(f"🔲 Patch Size: {official_distrifusion_model.patch_size}")
            if hasattr(official_distrifusion_model, 'split_mode'):
                info.append(f"✂️  Split Mode: {official_distrifusion_model.split_mode}")
            if hasattr(official_distrifusion_model, 'current_step'):
                info.append(f"🔄 Current Step: {official_distrifusion_model.current_step}")
            
            # GPU information
            if torch.cuda.is_available():
                info.append(f"🎮 CUDA Devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    info.append(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                info.append("⚠️ CUDA not available")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"❌ Error getting status: {e}"


# Node mappings
NODE_CLASS_MAPPINGS = {}

if OFFICIAL_DISTRIFUSION_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "OfficialDistriFusionModelLoader": OfficialDistriFusionModelLoader,
        "OfficialDistriFusionSampler": OfficialDistriFusionSampler,
        "OfficialDistriFusionStatus": OfficialDistriFusionStatus,
    })

NODE_DISPLAY_NAME_MAPPINGS = {}

if OFFICIAL_DISTRIFUSION_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "OfficialDistriFusionModelLoader": "Official DistriFusion Model Loader",
        "OfficialDistriFusionSampler": "Official DistriFusion Sampler",
        "OfficialDistriFusionStatus": "Official DistriFusion Status",
    }) 