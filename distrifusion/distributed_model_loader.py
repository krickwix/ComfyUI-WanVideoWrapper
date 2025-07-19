"""
Dedicated DistriFusion Model Loader for ComfyUI
Simplified and optimized model loading with automatic distribution setup
"""

import torch
import torch.distributed as dist
import os
import folder_paths
import comfy.model_management as mm

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
from .distrifusion_wrapper import create_distrifusion_model
from .patch_manager import PatchConfig, PatchSplitMode
from .communication import DistributedManager
from typing import Dict, Any, Tuple, Optional


class DistriFusionModelLoader:
    """
    Streamlined ComfyUI Node for loading WanVideo models with DistriFusion distribution
    Automatically handles multi-GPU setup and configuration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), 
                         {"tooltip": "WanVideo model to load with DistriFusion support"}),
                
                # Core model settings
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16", "tooltip": "Model precision"}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e5m2"], {"default": "disabled"}),
                
                # DistriFusion distribution settings
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "Number of GPUs for distribution"}),
                "split_strategy": (["auto", "spatial", "temporal", "spatiotemporal"], 
                                 {"default": "auto", "tooltip": "How to distribute the model across GPUs"}),
                "patch_size": ("INT", {"default": 64, "min": 32, "max": 128, "step": 8, "tooltip": "Base patch size for splitting"}),
                "overlap_ratio": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.025, "tooltip": "Overlap ratio for seamless boundaries"}),
                
                # Performance settings
                "async_comm": ("BOOLEAN", {"default": True, "tooltip": "Enable asynchronous communication"}),
                "warmup_steps": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Steps with full synchronization"}),
                "memory_optimization": (["balanced", "speed", "memory"], {"default": "balanced", "tooltip": "Optimization strategy"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa", "flash_attn_2", "flash_attn_3", "sageattn", "flex_attention"
                ], {"default": "sdpa"}),
                "force_rank": ("INT", {"default": -1, "min": -1, "max": 7, "tooltip": "Force specific GPU rank (-1 for auto)"}),
                "master_port": ("INT", {"default": 0, "min": 0, "max": 65535, "tooltip": "Master port (0 for auto)"}),
                "lora": ("WANVIDLORA", {"default": None}),
                "compile_model": ("BOOLEAN", {"default": False, "tooltip": "Compile model for better performance"}),
            }
        }

    RETURN_TYPES = ("DISTRIFUSION_MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load_model"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Load WanVideo model with automatic DistriFusion distribution setup"

    def load_model(self, model, precision, quantization, num_gpus, split_strategy, patch_size, overlap_ratio,
                   async_comm, warmup_steps, memory_optimization, attention_mode="sdpa", force_rank=-1, 
                   master_port=0, lora=None, compile_model=False):
        
        status_messages = []
        
        try:
            # 1. Validate GPU availability
            available_gpus = torch.cuda.device_count()
            if num_gpus > available_gpus:
                log.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available. Adjusting...")
                num_gpus = available_gpus
                status_messages.append(f"Adjusted to {num_gpus} GPUs (hardware limit)")
            
            # 2. Determine distribution strategy
            if num_gpus == 1:
                log.info("Single GPU mode - DistriFusion disabled")
                return self._load_single_gpu(model, precision, quantization, attention_mode, lora, status_messages)
            
            # 3. Auto-configure split strategy based on model and hardware
            actual_split_mode = self._determine_split_strategy(split_strategy, num_gpus, status_messages)
            
            # 4. Calculate optimal patch configuration
            patch_config = self._calculate_patch_config(
                actual_split_mode, num_gpus, patch_size, overlap_ratio, warmup_steps, async_comm
            )
            status_messages.append(f"Patch config: {actual_split_mode}, overlap={patch_config.patch_overlap}px")
            
            # 5. Setup distributed environment
            rank = self._setup_distributed_environment(num_gpus, force_rank, master_port, status_messages)
            
            # 6. Load base model with optimizations
            base_model = self._load_base_model(
                model, precision, quantization, attention_mode, memory_optimization, lora, status_messages
            )
            
            # 7. Create DistriFusion wrapper
            distrifusion_model = self._create_distrifusion_model(
                base_model, patch_config, num_gpus, rank, compile_model, status_messages
            )
            
            # 8. Final status
            status_messages.append("✅ DistriFusion model loaded successfully")
            status_messages.append(f"Ready for {num_gpus}x GPU inference")
            
            status_text = "\n".join(status_messages)
            return (distrifusion_model, status_text)
            
        except Exception as e:
            error_msg = f"❌ Failed to load DistriFusion model: {str(e)}"
            status_messages.append(error_msg)
            log.error(error_msg)
            raise RuntimeError("\n".join(status_messages))
    
    def _load_single_gpu(self, model, precision, quantization, attention_mode, lora, status_messages):
        """Load model for single GPU inference"""
        status_messages.append("Loading single GPU model...")
        
        # Check if WanVideoModelLoader is available
        if WanVideoModelLoader is None:
            raise ImportError(
                "WanVideoModelLoader is not available. Please ensure that the "
                "nodes_model_loading module is properly accessible."
            )
        
        # Use standard WanVideo loader
        loader = WanVideoModelLoader()
        wan_model = loader.loadmodel(
            model=model,
            base_precision=precision,
            quantization=quantization,
            load_device="main_device",
            attention_mode=attention_mode,
            lora=lora
        )[0]
        
        # Mark as non-distributed
        wan_model.model["distrifusion_enabled"] = False
        wan_model.model["num_gpus"] = 1
        
        status_messages.append("✅ Single GPU model loaded")
        return (wan_model, "\n".join(status_messages))
    
    def _determine_split_strategy(self, split_strategy, num_gpus, status_messages):
        """Automatically determine the best split strategy"""
        if split_strategy != "auto":
            return getattr(PatchSplitMode, split_strategy.upper() + "_ONLY", PatchSplitMode.SPATIAL_ONLY)
        
        # Auto-selection logic
        if num_gpus <= 4:
            # For 2-4 GPUs, spatial splitting is most efficient
            strategy = PatchSplitMode.SPATIAL_ONLY
            status_messages.append(f"Auto-selected: spatial splitting for {num_gpus} GPUs")
        elif num_gpus <= 6:
            # For 5-6 GPUs, hybrid approach
            strategy = PatchSplitMode.SPATIOTEMPORAL
            status_messages.append(f"Auto-selected: spatiotemporal splitting for {num_gpus} GPUs")
        else:
            # For 7+ GPUs, full temporal splitting
            strategy = PatchSplitMode.TEMPORAL_ONLY
            status_messages.append(f"Auto-selected: temporal splitting for {num_gpus} GPUs")
        
        return strategy
    
    def _calculate_patch_config(self, split_mode, num_gpus, patch_size, overlap_ratio, warmup_steps, async_comm):
        """Calculate optimal patch configuration"""
        overlap_pixels = max(4, int(patch_size * overlap_ratio))
        
        # Adjust temporal chunk size for temporal splitting
        temporal_chunk_size = None
        if split_mode in [PatchSplitMode.TEMPORAL_ONLY, PatchSplitMode.SPATIOTEMPORAL]:
            # Base chunk size, will be adjusted based on actual video length
            temporal_chunk_size = max(8, 64 // num_gpus)
        
        return PatchConfig(
            num_devices=num_gpus,
            split_mode=split_mode,
            patch_overlap=overlap_pixels,
            temporal_chunk_size=temporal_chunk_size,
            sync_first_step=True,
            async_boundary_update=async_comm,
            warmup_steps=warmup_steps
        )
    
    def _setup_distributed_environment(self, num_gpus, force_rank, master_port, status_messages):
        """Setup distributed environment with automatic configuration"""
        
        # Determine rank
        if force_rank >= 0:
            rank = force_rank
            status_messages.append(f"Using forced rank: {rank}")
        else:
            # Try to get rank from environment
            rank = int(os.environ.get('RANK', 0))
            if rank == 0 and 'LOCAL_RANK' in os.environ:
                rank = int(os.environ['LOCAL_RANK'])
            status_messages.append(f"Auto-detected rank: {rank}")
        
        # Setup distributed if not already initialized
        if num_gpus > 1 and not dist.is_initialized():
            try:
                # Auto-assign port if needed
                if master_port == 0:
                    master_port = self._find_free_port()
                
                os.environ.setdefault('MASTER_ADDR', 'localhost')
                os.environ.setdefault('MASTER_PORT', str(master_port))
                os.environ.setdefault('WORLD_SIZE', str(num_gpus))
                os.environ.setdefault('RANK', str(rank))
                
                dist.init_process_group(
                    backend='nccl',
                    rank=rank,
                    world_size=num_gpus
                )
                
                status_messages.append(f"Initialized distributed: rank {rank}/{num_gpus}")
                
            except Exception as e:
                status_messages.append(f"⚠️ Distributed init failed: {e}")
                log.warning(f"Failed to initialize distributed environment: {e}")
        
        return rank
    
    def _load_base_model(self, model, precision, quantization, attention_mode, memory_optimization, lora, status_messages):
        """Load the base WanVideo model with optimizations"""
        status_messages.append("Loading base WanVideo model...")
        
        # Determine load device based on memory optimization
        load_device = "main_device" if memory_optimization == "speed" else "offload_device"
        
        # Apply quantization if memory optimization is set to memory
        if memory_optimization == "memory" and quantization == "disabled":
            quantization = "fp8_e4m3fn"
            status_messages.append("Applied auto-quantization for memory optimization")
        
        # Check if WanVideoModelLoader is available
        if WanVideoModelLoader is None:
            raise ImportError(
                "WanVideoModelLoader is not available. Please ensure that the "
                "nodes_model_loading module is properly accessible."
            )
        
        loader = WanVideoModelLoader()
        wan_model_patcher = loader.loadmodel(
            model=model,
            base_precision=precision,
            quantization=quantization,
            load_device=load_device,
            attention_mode=attention_mode,
            lora=lora
        )[0]
        
        base_model = wan_model_patcher.model.diffusion_model
        status_messages.append(f"Base model loaded: {precision}, {quantization}")
        
        return base_model
    
    def _create_distrifusion_model(self, base_model, patch_config, num_gpus, rank, compile_model, status_messages):
        """Create the DistriFusion wrapper"""
        status_messages.append("Creating DistriFusion wrapper...")
        
        distrifusion_model = create_distrifusion_model(
            wan_model=base_model,
            num_devices=num_gpus,
            split_mode=patch_config.split_mode.value,
            patch_overlap=patch_config.patch_overlap,
            world_size=num_gpus,
            rank=rank
        )
        
        # Apply patch config
        distrifusion_model.patch_config = patch_config
        
        # Optional compilation for performance
        if compile_model:
            try:
                # Compile the underlying model for better performance
                distrifusion_model.wan_model = torch.compile(distrifusion_model.wan_model)
                status_messages.append("Model compiled for optimized performance")
            except Exception as e:
                status_messages.append(f"⚠️ Compilation failed: {e}")
        
        return distrifusion_model
    
    def _find_free_port(self):
        """Find a free port for distributed communication"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]


class DistriFusionDistributionConfig:
    """
    Node for configuring DistriFusion distribution parameters
    Can be used to preset distribution settings
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 8}),
                "split_mode": (["spatial", "temporal", "spatiotemporal"], {"default": "spatial"}),
                "communication_backend": (["nccl", "gloo"], {"default": "nccl"}),
                "sync_frequency": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Sync every N steps"}),
            },
            "optional": {
                "patch_overlap": ("INT", {"default": 8, "min": 0, "max": 32}),
                "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 10}),
                "async_communication": ("BOOLEAN", {"default": True}),
                "memory_efficient": ("BOOLEAN", {"default": False, "tooltip": "Use memory-efficient mode"}),
                "debug_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable debug logging"}),
            }
        }

    RETURN_TYPES = ("DISTRIFUSION_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "WanVideoWrapper/DistriFusion"
    DESCRIPTION = "Configure DistriFusion distribution parameters"

    def create_config(self, num_gpus, split_mode, communication_backend, sync_frequency,
                      patch_overlap=8, warmup_steps=4, async_communication=True, 
                      memory_efficient=False, debug_mode=False):
        
        # Create configuration dictionary
        config = {
            "num_gpus": num_gpus,
            "split_mode": split_mode,
            "backend": communication_backend,
            "sync_frequency": sync_frequency,
            "patch_overlap": patch_overlap,
            "warmup_steps": warmup_steps,
            "async_communication": async_communication,
            "memory_efficient": memory_efficient,
            "debug_mode": debug_mode,
            "patch_config": PatchConfig(
                num_devices=num_gpus,
                split_mode=getattr(PatchSplitMode, split_mode.upper() + "_ONLY", PatchSplitMode.SPATIAL_ONLY),
                patch_overlap=patch_overlap,
                warmup_steps=warmup_steps,
                async_boundary_update=async_communication
            )
        }
        
        if debug_mode:
            # Enable debug logging
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
            os.environ['NCCL_DEBUG'] = 'INFO'
        
        return (config,)


# Register the new nodes
NODE_CLASS_MAPPINGS = {
    "DistriFusionModelLoader": DistriFusionModelLoader,
    "DistriFusionDistributionConfig": DistriFusionDistributionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DistriFusionModelLoader": "DistriFusion Model Loader",
    "DistriFusionDistributionConfig": "DistriFusion Config",
} 