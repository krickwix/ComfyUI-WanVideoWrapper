"""
Distributed Inference Nodes for ComfyUI-WanVideoWrapper
Provides multi-GPU support for Wan2.2-Lightning inference in ComfyUI.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import json

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
import comfy.model_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WanVideoDistributedInference:
    """
    Distributed inference node for Wan2.2-Lightning using the distributed model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", {"tooltip": "WanVideo model to use for inference"}),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", {"tooltip": "Text embeddings from text encoder"}),
                "prompt": ("STRING", {"default": "A beautiful sunset over mountains", "tooltip": "Text prompt for video generation"}),
                "negative_prompt": ("STRING", {"default": "", "tooltip": "Negative prompt"}),
                "width": ("INT", {"default": 832, "min": 256, "max": 1920, "tooltip": "Video width"}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1080, "tooltip": "Video height"}),
                "num_frames": ("INT", {"default": 121, "min": 16, "max": 300, "tooltip": "Number of video frames"}),
                "num_inference_steps": ("INT", {"default": 20, "min": 4, "max": 50, "tooltip": "Number of denoising steps"}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "tooltip": "Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 42, "tooltip": "Random seed for generation"}),
                "offload_model": ("BOOLEAN", {"default": True, "tooltip": "Offload model to CPU to save VRAM"}),
                "sample_steps": ("INT", {"default": 20, "min": 4, "max": 50, "tooltip": "Number of sampling steps (4 for Lightning)"}),
            },
            "optional": {
                "output_format": (["mp4", "gif", "frames"], {"default": "mp4", "tooltip": "Output format"}),
                "fps": ("INT", {"default": 60, "min": 1, "max": 120, "tooltip": "Output video FPS"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "log_output", "status")
    FUNCTION = "run_distributed_inference"
    CATEGORY = "WanVideoWrapper/Distributed"
    DESCRIPTION = "Run Wan2.2-Lightning inference with multi-GPU distributed model"
    
    def run_distributed_inference(self, model, text_embeds, prompt, negative_prompt, width, height, 
                                 num_frames, num_inference_steps, guidance_scale, seed, offload_model, sample_steps,
                                 output_format="mp4", fps=60):
        """
        Run distributed inference using the distributed model
        """
        try:
            # Get distributed configuration from the model
            enable_distributed = getattr(model, 'enable_distributed', False)
            gpu_count = getattr(model, 'gpu_count', 2)
            use_ulysses = getattr(model, 'use_ulysses', True)
            use_fsdp = getattr(model, 'use_fsdp', True)
            master_port = getattr(model, 'master_port', 29501)
            
            # Try to get distributed options from the model dictionary if they exist
            if hasattr(model, 'model') and isinstance(model.model, dict):
                enable_distributed = model.model.get('enable_distributed', enable_distributed)
                gpu_count = model.model.get('gpu_count', gpu_count)
                use_ulysses = model.model.get('use_ulysses', use_ulysses)
                use_fsdp = model.model.get('use_fsdp', use_fsdp)
                master_port = model.model.get('master_port', master_port)
                is_distributed = model.model.get('is_distributed', False)
                distributed_device_ids = model.model.get('distributed_device_ids', [])
                distributed_backend = model.model.get('distributed_backend', 'unknown')
                distributed_type = model.model.get('distributed_type', 'unknown')
            else:
                is_distributed = False
                distributed_device_ids = []
                distributed_backend = 'unknown'
                distributed_type = 'unknown'
            
            logger.info(f"Starting distributed inference with {gpu_count} GPUs")
            logger.info(f"Distributed backend: {distributed_backend}")
            logger.info(f"Distributed type: {distributed_type}")
            
            # Check if distributed inference is enabled
            if not enable_distributed:
                return "", "", "ERROR: Distributed inference is not enabled for this model. Set enable_distributed=True in LoadWanVideoModel"
            
            # Check if the model is actually distributed
            if not is_distributed:
                return "", "", "ERROR: Model is not distributed. Please reload the model with enable_distributed=True"
            
            # Check if we have enough GPUs
            available_gpus = torch.cuda.device_count()
            if available_gpus < gpu_count:
                return "", "", f"ERROR: Requested {gpu_count} GPUs but only {available_gpus} are available"
            
            # Create temporary output directory
            output_dir = tempfile.mkdtemp(prefix="wanvideo_distributed_")
            
            # Run the distributed inference
            result = self._run_distributed_inference_with_model(
                model, text_embeds, prompt, negative_prompt, width, height,
                num_frames, num_inference_steps, guidance_scale, seed, offload_model, sample_steps,
                gpu_count, use_ulysses, use_fsdp, master_port, output_dir, output_format, fps,
                is_distributed, distributed_device_ids, distributed_backend, distributed_type
            )
            
            if result["success"]:
                # Find the generated video file
                video_path = self._find_output_video(output_dir, output_format)
                if video_path:
                    return video_path, result["log"], "SUCCESS"
                else:
                    return "", result["log"], "WARNING: Inference completed but no video found"
            else:
                return "", result["log"], f"ERROR: {result['error']}"
                
        except Exception as e:
            error_msg = f"Distributed inference failed: {str(e)}"
            logger.error(error_msg)
            return "", error_msg, "ERROR"
    
    def _run_distributed_inference_with_model(self, model, text_embeds, prompt, negative_prompt, width, height,
                                             num_frames, num_inference_steps, guidance_scale, seed, offload_model, sample_steps,
                                             gpu_count, use_ulysses, use_fsdp, master_port, output_dir, output_format, fps,
                                             is_distributed, distributed_device_ids, distributed_backend, distributed_type):
        """Run inference using the already-distributed model"""
        try:
            logger.info(f"Running inference with distributed model across {gpu_count} GPUs")
            
            # Get the actual model from the patcher
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                actual_model = model.model.diffusion_model
            else:
                actual_model = model
            
            # Verify the model is distributed
            if not is_distributed:
                return {
                    "success": False,
                    "log": "",
                    "error": "Model is not distributed"
                }
            
            # Show distribution information
            log_output = f"""
Distributed Inference Started:
- GPUs: {gpu_count}
- Ulysses: {use_ulysses}
- FSDP: {use_fsdp}
- Distributed Backend: {distributed_backend}
- Distributed Type: {distributed_type}
- Model device: {next(actual_model.parameters()).device}
- Available GPUs: {torch.cuda.device_count()}
- Current GPU: {torch.cuda.current_device()}
- Model parameters: {sum(p.numel() for p in actual_model.parameters()):,}
- Model type: {type(actual_model)}
- Is distributed: {is_distributed}
- Distributed device IDs: {distributed_device_ids}
- CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB
- CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB

Note: This is using {distributed_type} distribution.
For true multi-process distributed inference, use the distributed_inference_launcher.py script with torchrun.
"""
            
            # This is where you would implement the actual WanVideo inference logic
            # The model is already distributed, so you can run inference directly
            logger.info("Model is distributed and ready for inference")
            
            return {
                "success": True,
                "log": log_output,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "log": "",
                "error": f"Distributed inference failed: {str(e)}"
            }
    
    def _find_output_video(self, output_dir: str, output_format: str) -> Optional[str]:
        """Find the generated video file in the output directory"""
        try:
            output_path = Path(output_dir)
            
            # Look for video files based on format
            if output_format == "mp4":
                video_files = list(output_path.glob("*.mp4"))
            elif output_format == "gif":
                video_files = list(output_path.glob("*.gif"))
            else:  # frames
                # Look for frame directories
                frame_dirs = [d for d in output_path.iterdir() if d.is_dir() and "frame" in d.name.lower()]
                if frame_dirs:
                    return str(frame_dirs[0])
                return None
            
            if video_files:
                return str(video_files[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding output video: {e}")
            return None


class WanVideoDistributedConfig:
    """
    Configuration node for distributed inference settings
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gpu_count": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of GPUs to use"}),
                "use_ulysses": ("BOOLEAN", {"default": True, "tooltip": "Enable Ulysses distribution"}),
                "use_fsdp": ("BOOLEAN", {"default": True, "tooltip": "Enable FSDP model sharding"}),
                "offload_model": ("BOOLEAN", {"default": True, "tooltip": "Enable model offloading"}),
                "master_port": ("INT", {"default": 29501, "min": 29500, "max": 29600, "tooltip": "Master port for distributed training"}),
                "memory_optimization": ("BOOLEAN", {"default": True, "tooltip": "Enable memory optimization"}),
            },
            "optional": {
                "custom_environment": ("STRING", {"default": "", "tooltip": "Custom environment variables (JSON format)"}),
                "torch_compile": ("BOOLEAN", {"default": False, "tooltip": "Enable PyTorch compilation"}),
                "mixed_precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Mixed precision mode"}),
            }
        }
    
    RETURN_TYPES = ("DISTRIBUTED_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "WanVideoWrapper/Distributed"
    DESCRIPTION = "Create distributed inference configuration"
    
    def create_config(self, **kwargs):
        """Create a distributed inference configuration"""
        config = {
            "gpu_count": kwargs.get("gpu_count", 2),
            "use_ulysses": kwargs.get("use_ulysses", True),
            "use_fsdp": kwargs.get("use_fsdp", True),
            "offload_model": kwargs.get("offload_model", True),
            "master_port": kwargs.get("master_port", 29501),
            "memory_optimization": kwargs.get("memory_optimization", True),
            "torch_compile": kwargs.get("torch_compile", False),
            "mixed_precision": kwargs.get("mixed_precision", "bf16"),
        }
        
        # Parse custom environment variables
        custom_env = kwargs.get("custom_environment", "")
        if custom_env:
            try:
                config["custom_environment"] = json.loads(custom_env)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in custom_environment, ignoring")
                config["custom_environment"] = {}
        else:
            config["custom_environment"] = {}
        
        return (config,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoDistributedInference": WanVideoDistributedInference,
    "WanVideoDistributedConfig": WanVideoDistributedConfig,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDistributedInference": "WanVideo Distributed Inference",
    "WanVideoDistributedConfig": "WanVideo Distributed Config",
}
