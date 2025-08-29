"""
Distributed Inference Nodes for ComfyUI-WanVideoWrapper
Provides multi-GPU Ulysses distribution support for Wan2.2-Lightning inference in ComfyUI.
"""

import os
import sys
import torch
import subprocess
import tempfile
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
import comfy.model_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WanVideoDistributedInference:
    """
    Distributed inference node for Wan2.2-Lightning using Ulysses distribution
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
                "custom_script_path": ("STRING", {"default": "", "tooltip": "Custom path to generate.py script"}),
                "model_cache_dir": ("STRING", {"default": "", "tooltip": "Custom model cache directory"}),
                "output_format": (["mp4", "gif", "frames"], {"default": "mp4", "tooltip": "Output format"}),
                "fps": ("INT", {"default": 60, "min": 1, "max": 120, "tooltip": "Output video FPS"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "log_output", "status")
    FUNCTION = "run_distributed_inference"
    CATEGORY = "WanVideoWrapper/Distributed"
    DESCRIPTION = "Run Wan2.2-Lightning inference with multi-GPU Ulysses distribution"
    
    def run_distributed_inference(self, model, text_embeds, prompt, negative_prompt, width, height, 
                                 num_frames, num_inference_steps, guidance_scale, seed, offload_model, sample_steps,
                                 custom_script_path="", model_cache_dir="", output_format="mp4", fps=60):
        """
        Run distributed inference using Ulysses distribution
        """
        try:
            logger.info(f"Starting distributed inference with {gpu_count} GPUs")
            
            # Get model path from the loaded model
            model_path = self._get_model_path_from_model(model)
            if not model_path:
                return "", "", "ERROR: Could not determine model path from loaded model"
            
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
            
            # Check if distributed inference is enabled
            if not enable_distributed:
                return "", "", "ERROR: Distributed inference is not enabled for this model. Set enable_distributed=True in LoadWanVideoModel"
            
            # Create temporary output directory
            output_dir = tempfile.mkdtemp(prefix="wanvideo_distributed_")
            
            # Build the inference command
            cmd = self._build_inference_command(
                model_path=model_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                gpu_count=gpu_count,
                use_ulysses=use_ulysses,
                use_fsdp=use_fsdp,
                offload_model=offload_model,
                sample_steps=sample_steps,
                output_dir=output_dir,
                custom_script_path=custom_script_path,
                model_cache_dir=model_cache_dir,
                output_format=output_format,
                fps=fps
            )
            
            # Run the distributed inference
            result = self._execute_distributed_inference(cmd, gpu_count)
            
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
    
    def _get_model_path_from_model(self, model) -> Optional[str]:
        """Extract model path from the loaded WanVideo model"""
        try:
            # Try to get the model path from the model object
            if hasattr(model, 'model_path'):
                return model.model_path
            
            # Check if it's a WanVideoModel with base_path info
            if hasattr(model, 'base_path'):
                return model.base_path
            
            # Check if it's a WanVideoModel with model_name info
            if hasattr(model, 'model_name'):
                model_name = model.model_name
                # Try to get the full path from folder_paths
                try:
                    return folder_paths.get_full_path("diffusion_models", model_name)
                except:
                    # If that fails, try to construct the path manually
                    models_dir = folder_paths.models_dir
                    return os.path.join(models_dir, "diffusion_models", model_name)
            
            # Check if it's a WanVideoModel with pipeline info
            if hasattr(model, 'pipeline') and 'model_path' in model.pipeline:
                return model.pipeline['model_path']
            
            # Try to get from folder_paths
            model_name = getattr(model, 'model_name', None)
            if model_name:
                try:
                    return folder_paths.get_full_path("diffusion_models", model_name)
                except:
                    # If that fails, try to construct the path manually
                    models_dir = folder_paths.models_dir
                    return os.path.join(models_dir, "diffusion_models", model_name)
            
            logger.warning("Could not determine model path from model object")
            return None
            
        except Exception as e:
            logger.error(f"Error getting model path: {e}")
            return None
    
    def _build_inference_command(self, **kwargs) -> List[str]:
        """Build the inference command with all parameters"""
        cmd = []
        
        # Set environment variables
        env_vars = {
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(kwargs['gpu_count'])),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
        }
        
        # Add environment variables to command
        for key, value in env_vars.items():
            cmd.extend([f"{key}={value}"])
        
        # Add torchrun for distributed training
        if kwargs['gpu_count'] > 1:
            cmd.extend([
                "torchrun",
                f"--nproc_per_node={kwargs['gpu_count']}",
                "--master_port=29501"  # Avoid port conflicts
            ])
        
        # Add the generate.py script path
        script_path = kwargs.get('custom_script_path') or "/Wan2.2/generate.py"
        cmd.append(script_path)
        
        # Add basic parameters
        cmd.extend([
            "--task", "t2v-A14B",
            "--size", f"{kwargs['width']}*{kwargs['height']}",
            "--ckpt_dir", kwargs['model_path'],
            "--base_seed", str(kwargs['seed']),
            "--prompt", kwargs['prompt'],
            "--frame_num", str(kwargs['num_frames']),
            "--sample_steps", str(kwargs['sample_steps'])
        ])
        
        # Add negative prompt if provided
        if kwargs.get('negative_prompt'):
            cmd.extend(["--negative_prompt", kwargs['negative_prompt']])
        
        # Add guidance scale
        if kwargs.get('guidance_scale'):
            cmd.extend(["--guidance_scale", str(kwargs['guidance_scale'])])
        
        # Add model offloading
        if kwargs.get('offload_model'):
            cmd.append("--offload_model")
        
        # Add Ulysses distribution parameters
        if kwargs['gpu_count'] > 1 and kwargs.get('use_ulysses'):
            cmd.extend([
                f"--ulysses_size", str(kwargs['gpu_count'])
            ])
            
            # Add FSDP if enabled
            if kwargs.get('use_fsdp'):
                cmd.extend(["--dit_fsdp", "--t5_fsdp"])
        
        # Add output directory
        cmd.extend(["--output_dir", kwargs['output_dir']])
        
        # Add custom model cache if specified
        if kwargs.get('model_cache_dir'):
            cmd.extend(["--model_cache_dir", kwargs['model_cache_dir']])
        
        logger.info(f"Built command: {' '.join(cmd)}")
        return cmd
    
    def _execute_distributed_inference(self, cmd: List[str], gpu_count: int) -> Dict[str, Any]:
        """Execute the distributed inference command"""
        try:
            logger.info(f"Executing distributed inference with {gpu_count} GPUs")
            
            # Set working directory to Wan2.2
            working_dir = "/Wan2.2"
            if not os.path.exists(working_dir):
                working_dir = os.getcwd()
            
            # Execute the command
            process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Collect output in real-time
            stdout_lines = []
            stderr_lines = []
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    stdout_lines.append(output.strip())
                    logger.info(output.strip())
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            stdout_lines.extend(stdout.strip().split('\n') if stdout else [])
            stderr_lines.extend(stderr.strip().split('\n') if stderr else [])
            
            # Check if process was successful
            if process.returncode == 0:
                return {
                    "success": True,
                    "log": "\n".join(stdout_lines),
                    "error": None
                }
            else:
                error_log = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
                return {
                    "success": False,
                    "log": "\n".join(stdout_lines),
                    "error": error_log
                }
                
        except Exception as e:
            return {
                "success": False,
                "log": "",
                "error": f"Execution failed: {str(e)}"
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
