import torch
import os
import gc
from .utils import log
import numpy as np
from tqdm import tqdm

from .wanvideo.modules.model import WanModel
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.modules.clip import CLIPModel

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import accelerate

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import comfy.model_base
from comfy.sd import load_lora_for_models

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

try:
    from server import PromptServer
except:
    PromptServer = None

# Import Wan2.1 distributed components
try:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    from functools import partial
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    log.warning("FSDP not available. Please install PyTorch with distributed support.")

try:
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        get_sp_group,
    )
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    XFUSER_AVAILABLE = True
except ImportError:
    XFUSER_AVAILABLE = False
    log.warning("xfuser not available. Please install xfuser for context parallel support.")

class WanDistributedConfig:
    """Configuration class for Wan2.1 distributed inference settings"""
    def __init__(self, 
                 world_size=1,
                 rank=0,
                 backend="nccl",
                 init_method="env://",
                 use_fsdp=False,
                 use_context_parallel=False,
                 ulysses_size=1,
                 ring_size=1,
                 t5_fsdp=False,
                 dit_fsdp=False,
                 use_usp=False,
                 t5_cpu=False,
                 param_dtype=torch.bfloat16,
                 reduce_dtype=torch.float32,
                 buffer_dtype=torch.float32,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 sync_module_states=True):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.init_method = init_method
        self.use_fsdp = use_fsdp
        self.use_context_parallel = use_context_parallel
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.buffer_dtype = buffer_dtype
        self.sharding_strategy = sharding_strategy
        self.sync_module_states = sync_module_states

# Import the regular WanVideoModel for compatibility
from .nodes_model_loading import WanVideoModel, WanVideoModelConfig

class WanDistributedModel(WanVideoModel):
    """WanVideo model wrapper for Wan2.1 distributed inference"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributed_config = None
        self.fsdp_model = None
        self.is_distributed = False



    def setup_distributed_inference(self, config: WanDistributedConfig):
        """Setup Wan2.1 distributed inference"""
        log.info("🌐 Starting distributed inference setup...")
        
        if not FSDP_AVAILABLE:
            log.error("❌ FSDP not available!")
            raise ImportError("FSDP is required for distributed inference")
        log.info("✅ FSDP is available")
        
        self.distributed_config = config
        self.is_distributed = False
        
        log.info(f"📋 Distributed config:")
        log.info(f"   - World size: {config.world_size}")
        log.info(f"   - Rank: {config.rank}")
        log.info(f"   - Backend: {config.backend}")
        log.info(f"   - Use FSDP: {config.use_fsdp}")
        log.info(f"   - Use context parallel: {config.use_context_parallel}")
        log.info(f"   - Ulysses size: {config.ulysses_size}")
        log.info(f"   - Ring size: {config.ring_size}")
        
        # Only initialize distributed environment if actually using multiple GPUs
        if config.world_size > 1:
            log.info("🚀 Initializing multi-GPU distributed environment...")
            try:
                # For single-server multi-GPU, we'll use a simplified approach
                # Instead of full distributed processes, we'll use DataParallel-like sharding
                log.info("🔄 Single-server multi-GPU mode detected")
                log.info("   - Using simplified multi-GPU approach (no process spawning)")
                
                # Set up device mapping for multi-GPU
                self.device_map = {}
                self.gpu_devices = list(range(config.world_size))
                log.info(f"   - GPU devices: {self.gpu_devices}")
                
                # Mark as distributed but with simplified setup
                self.is_distributed = True
                self.is_single_server = True
                log.info(f"✅ Initialized single-server multi-GPU environment with {config.world_size} GPUs")
                
                # For single-server, context parallel requires process spawning which we're avoiding
                if config.use_context_parallel:
                    log.warning("⚠️ Context parallel disabled for single-server mode (requires process spawning)")
                    log.info("   - Use FSDP or device sharding instead for multi-GPU acceleration")
                    config.use_context_parallel = False
                else:
                    log.info("ℹ️ Context parallel not enabled")
            except Exception as e:
                log.error(f"❌ Failed to initialize distributed environment: {e}")
                log.info("🔄 Falling back to single-GPU mode")
                config.world_size = 1
                config.rank = 0
                self.is_distributed = False
        else:
            log.info("ℹ️ Single-GPU mode, skipping distributed initialization")
        
        # Apply multi-GPU sharding if enabled (only for multi-GPU)
        if config.world_size > 1:
            log.info("🔧 Applying multi-GPU sharding...")
            try:
                if config.use_fsdp and hasattr(self, 'is_single_server') and self.is_single_server:
                    # For single-server, use device sharding instead of FSDP
                    log.info("   - Single-server mode: using device sharding instead of FSDP")
                    self._apply_device_sharding(config)
                    log.info("✅ Applied device sharding to model")
                elif config.use_fsdp:
                    # Use full FSDP for multi-server
                    self.fsdp_model = self._apply_fsdp(config)
                    log.info("✅ Applied full FSDP to model")
                else:
                    # Use simple device sharding
                    self._apply_device_sharding(config)
                    log.info("✅ Applied device sharding to model")
            except Exception as e:
                log.error(f"❌ Failed to apply multi-GPU sharding: {e}")
                log.info("🔄 Falling back to single-GPU mode")
                config.world_size = 1
                config.rank = 0
                self.is_distributed = False
        else:
            log.info("ℹ️ Single-GPU mode, no sharding needed")
        
        log.info("=" * 60)
        log.info("🎯 DISTRIBUTED INFERENCE SETUP COMPLETE")
        log.info("=" * 60)
        log.info(f"📊 Final configuration:")
        log.info(f"   - FSDP: {config.use_fsdp}")
        log.info(f"   - Context Parallel: {config.use_context_parallel}")
        log.info(f"   - World Size: {config.world_size}")
        log.info(f"   - Is Distributed: {self.is_distributed}")
        log.info("=" * 60)

    def _apply_fsdp(self, config):
        """Apply FSDP to the model"""
        log.info("🔧 Applying FSDP to model...")
        
        if not hasattr(self, 'diffusion_model') or self.diffusion_model is None:
            log.error("❌ Model must be loaded before applying FSDP")
            raise ValueError("Model must be loaded before applying FSDP")
        
        log.info(f"   - Param dtype: {config.param_dtype}")
        log.info(f"   - Reduce dtype: {config.reduce_dtype}")
        log.info(f"   - Buffer dtype: {config.buffer_dtype}")
        log.info(f"   - Sharding strategy: {config.sharding_strategy}")
        log.info(f"   - Sync module states: {config.sync_module_states}")
        log.info(f"   - Current device: {torch.cuda.current_device()}")
        
        # Define auto wrap policy for transformer blocks
        def auto_wrap_policy(module):
            return module in self.diffusion_model.blocks
        
        log.info("   - Auto wrap policy: wrap transformer blocks")
        
        try:
            # Apply FSDP
            fsdp_model = FSDP(
                module=self.diffusion_model,
                process_group=None,  # Use default process group
                sharding_strategy=config.sharding_strategy,
                auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=auto_wrap_policy),
                mixed_precision=MixedPrecision(
                    param_dtype=config.param_dtype,
                    reduce_dtype=config.reduce_dtype,
                    buffer_dtype=config.buffer_dtype
                ),
                device_id=torch.cuda.current_device(),
                sync_module_states=config.sync_module_states
            )
            
            log.info("✅ FSDP applied successfully")
            return fsdp_model
        except Exception as e:
            log.error(f"❌ Failed to apply FSDP: {e}")
            raise

    def _apply_single_server_fsdp(self, config):
        """Apply simplified FSDP for single-server multi-GPU"""
        log.info("🔧 Applying single-server FSDP...")
        
        if not hasattr(self, 'diffusion_model') or self.diffusion_model is None:
            log.error("❌ Model must be loaded before applying FSDP")
            raise ValueError("Model must be loaded before applying FSDP")
        
        log.info(f"   - Using {len(self.gpu_devices)} GPUs: {self.gpu_devices}")
        log.info(f"   - Param dtype: {config.param_dtype}")
        log.info(f"   - Sharding strategy: {config.sharding_strategy}")
        
        try:
            # Initialize a simple process group for single-server FSDP
            log.info("   - Initializing single-server process group...")
            
            # Set environment variables for single-node setup
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # Initialize process group with single process
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=config.backend,
                    init_method="env://",
                    rank=0,
                    world_size=1
                )
                log.info("   - Process group initialized for single-server FSDP")
            
            # Create FSDP wrapper
            fsdp_model = FSDP(
                module=self.diffusion_model,
                process_group=None,  # Use default process group
                sharding_strategy=config.sharding_strategy,
                auto_wrap_policy=None,  # No auto wrap for simplicity
                mixed_precision=MixedPrecision(
                    param_dtype=config.param_dtype,
                    reduce_dtype=config.reduce_dtype,
                    buffer_dtype=config.buffer_dtype
                ),
                device_id=torch.cuda.current_device(),
                sync_module_states=config.sync_module_states
            )
            
            log.info("✅ Single-server FSDP applied successfully")
            return fsdp_model
        except Exception as e:
            log.error(f"❌ Failed to apply single-server FSDP: {e}")
            raise

    def _apply_device_sharding(self, config):
        """Apply comprehensive device sharding across GPUs"""
        log.info("🔧 Applying device sharding...")
        
        if not hasattr(self, 'diffusion_model') or self.diffusion_model is None:
            log.error("❌ Model must be loaded before applying device sharding")
            raise ValueError("Model must be loaded before applying device sharding")
        
        log.info(f"   - Sharding across {len(self.gpu_devices)} GPUs: {self.gpu_devices}")
        
        try:
            model = self.diffusion_model
            
            # Verify model has required attributes
            if not hasattr(model, 'blocks'):
                log.error("❌ Model does not have 'blocks' attribute")
                raise ValueError("Model does not have required 'blocks' attribute")
            
            log.info(f"   - Model type: {type(model)}")
            log.info(f"   - Model has blocks: {hasattr(model, 'blocks')}")
            log.info(f"   - Number of blocks: {len(model.blocks) if hasattr(model, 'blocks') and model.blocks is not None else 'None'}")
            
            # Create DataParallel wrapper for multi-GPU inference
            log.info("🔧 Setting up DataParallel for multi-GPU inference...")
            self.parallel_model = torch.nn.DataParallel(model, device_ids=self.gpu_devices)
            log.info(f"✅ DataParallel setup complete with devices: {self.gpu_devices}")
            
            # Shard transformer blocks across GPUs (most memory intensive)
            if hasattr(model, 'blocks') and model.blocks is not None and len(model.blocks) > 0:
                blocks_per_gpu = max(1, len(model.blocks) // len(self.gpu_devices))
                log.info(f"   - Sharding {len(model.blocks)} blocks across {len(self.gpu_devices)} GPUs")
                log.info(f"   - Blocks per GPU: {blocks_per_gpu}")
                
                successful_blocks = 0
                for i, block in enumerate(model.blocks):
                    try:
                        if block is None:
                            log.warning(f"   - Block {i} is None, skipping")
                            continue
                            
                        gpu_idx = i // blocks_per_gpu
                        if gpu_idx >= len(self.gpu_devices):
                            gpu_idx = len(self.gpu_devices) - 1
                        device = f"cuda:{self.gpu_devices[gpu_idx]}"
                        block.to(device)
                        successful_blocks += 1
                        
                        if i % 5 == 0:  # Log every 5th block to avoid spam
                            log.info(f"   - Block {i} -> {device}")
                    except Exception as e:
                        log.warning(f"   - Failed to move block {i} to {device}: {e}")
                        # Move to first GPU as fallback
                        try:
                            block.to(f"cuda:{self.gpu_devices[0]}")
                            successful_blocks += 1
                            log.info(f"   - Block {i} -> cuda:{self.gpu_devices[0]} (fallback)")
                        except Exception as e2:
                            log.error(f"   - Failed to move block {i} to fallback device: {e2}")
                
                log.info(f"   - Successfully moved {successful_blocks}/{len(model.blocks)} blocks")
            else:
                log.warning("   - No transformer blocks found or blocks is None")
            
            # Distribute other components across GPUs
            components = []
            if hasattr(model, 'patch_embedding') and model.patch_embedding is not None:
                components.append(('patch_embedding', model.patch_embedding))
            if hasattr(model, 'pos_embedding') and model.pos_embedding is not None:
                components.append(('pos_embedding', model.pos_embedding))
            if hasattr(model, 'output_proj') and model.output_proj is not None:
                components.append(('output_proj', model.output_proj))
            if hasattr(model, 'add_conv_in') and model.add_conv_in is not None:
                components.append(('add_conv_in', model.add_conv_in))
            if hasattr(model, 'add_proj') and model.add_proj is not None:
                components.append(('add_proj', model.add_proj))
            if hasattr(model, 'attn_conv_in') and model.attn_conv_in is not None:
                components.append(('attn_conv_in', model.attn_conv_in))
            if hasattr(model, 'ref_conv') and model.ref_conv is not None:
                components.append(('ref_conv', model.ref_conv))
            if hasattr(model, 'control_adapter') and model.control_adapter is not None:
                components.append(('control_adapter', model.control_adapter))
            
            log.info(f"   - Found {len(components)} components to distribute")
            
            # Distribute components across GPUs
            for i, (name, component) in enumerate(components):
                try:
                    gpu_idx = i % len(self.gpu_devices)
                    device = f"cuda:{self.gpu_devices[gpu_idx]}"
                    component.to(device)
                    log.info(f"   - {name} -> {device}")
                except Exception as e:
                    log.warning(f"   - Failed to move {name} to {device}: {e}")
                    # Move to first GPU as fallback
                    try:
                        component.to(f"cuda:{self.gpu_devices[0]}")
                        log.info(f"   - {name} -> cuda:{self.gpu_devices[0]} (fallback)")
                    except Exception as e2:
                        log.error(f"   - Failed to move {name} to fallback device: {e2}")
            
            # Set the main device for the model
            model.main_device = f"cuda:{self.gpu_devices[0]}"
            log.info(f"   - Main device set to: {model.main_device}")
            
            # Override the model's forward method to use DataParallel
            original_forward = model.forward
            def parallel_forward(*args, **kwargs):
                # Ensure input is on the correct device
                x = args[0] if len(args) > 0 else kwargs.get('x')
                if x is not None:
                    x = x.to(f"cuda:{self.gpu_devices[0]}")
                    if len(args) > 0:
                        args = list(args)
                        args[0] = x
                    if 'x' in kwargs:
                        kwargs['x'] = x
                
                # Use DataParallel for inference
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    return self.parallel_model(*args, **kwargs)
            
            model.forward = parallel_forward
            log.info("✅ Model forward method overridden to use DataParallel")
            
            log.info("✅ Device sharding applied successfully")
            log.info(f"   - Model distributed across {len(self.gpu_devices)} GPUs")
            
        except Exception as e:
            log.error(f"❌ Failed to apply device sharding: {e}")
            raise

    def forward_distributed(self, *args, **kwargs):
        """Forward pass using distributed inference"""
        # Ensure inputs are in the correct dtype
        if hasattr(self, 'base_dtype'):
            args = [arg.to(dtype=self.base_dtype) if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.to(dtype=self.base_dtype) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        if self.fsdp_model is not None:
            return self.fsdp_model(*args, **kwargs)
        elif hasattr(self, 'diffusion_model') and hasattr(self, 'is_single_server') and self.is_single_server:
            # Use multi-GPU parallel forward pass for single-server mode
            return self._forward_multi_gpu(*args, **kwargs)
        elif hasattr(self, 'diffusion_model'):
            return self.diffusion_model(*args, **kwargs)
        else:
            raise ValueError("No model available for inference")
    
    def _forward_multi_gpu(self, *args, **kwargs):
        """Multi-GPU parallel forward pass for single-server mode"""
        if not hasattr(self, 'diffusion_model') or not hasattr(self, 'gpu_devices'):
            return self.diffusion_model(*args, **kwargs)
        
        model = self.diffusion_model
        
        # Use DataParallel for simple multi-GPU inference
        # This will automatically split the input across available GPUs
        if not hasattr(self, '_parallel_model'):
            log.info("🔧 Setting up DataParallel for multi-GPU inference...")
            self._parallel_model = torch.nn.DataParallel(model, device_ids=self.gpu_devices)
            log.info(f"✅ DataParallel setup complete with devices: {self.gpu_devices}")
        
        # Ensure input is on the correct device
        x = args[0] if len(args) > 0 else kwargs.get('x')
        if x is not None:
            # Move input to first GPU
            x = x.to(f"cuda:{self.gpu_devices[0]}")
            if len(args) > 0:
                args = list(args)
                args[0] = x
            if 'x' in kwargs:
                kwargs['x'] = x
        
        # Run inference using DataParallel
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = self._parallel_model(*args, **kwargs)
        
        return output

    def cleanup(self):
        """Cleanup distributed resources"""
        log.info("🧹 Cleaning up distributed resources...")
        
        if self.fsdp_model is not None:
            log.info("   - Cleaning up FSDP model...")
            try:
                # Free FSDP storage
                for m in self.fsdp_model.modules():
                    if isinstance(m, FSDP):
                        from torch.distributed.utils import _free_storage
                        _free_storage(m._handle.flat_param.data)
                del self.fsdp_model
                self.fsdp_model = None
                log.info("   - FSDP model cleaned up")
            except Exception as e:
                log.warning(f"   - FSDP cleanup warning: {e}")
        
        if self.is_distributed and hasattr(self, 'is_single_server') and self.is_single_server:
            log.info("   - Single-server mode, no process group to destroy")
            self.is_distributed = False
        elif self.is_distributed and dist.is_initialized():
            log.info("   - Destroying process group...")
            try:
                dist.destroy_process_group()
                self.is_distributed = False
                log.info("   - Process group destroyed")
            except Exception as e:
                log.warning(f"   - Process group cleanup warning: {e}")
        
        log.info("   - Running garbage collection...")
        gc.collect()
        torch.cuda.empty_cache()
        log.info("✅ Cleanup complete")

class WanDistributedConfigNode:
    """ComfyUI node for configuring Wan2.1 distributed settings"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Number of processes/GPUs (1 for single GPU)"}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1, "tooltip": "Process rank (0 to world_size-1)"}),
                "backend": (["nccl", "gloo"], {"default": "nccl", "tooltip": "Distributed backend"}),
                "use_fsdp": ("BOOLEAN", {"default": False, "tooltip": "Use FSDP for model sharding (requires world_size > 1)"}),
                "use_context_parallel": ("BOOLEAN", {"default": False, "tooltip": "Use context parallel (requires xfuser)"}),
                "ulysses_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Ulysses size for context parallel"}),
                "ring_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Ring size for context parallel"}),
                "t5_fsdp": ("BOOLEAN", {"default": False, "tooltip": "Use FSDP for T5 text encoder"}),
                "dit_fsdp": ("BOOLEAN", {"default": False, "tooltip": "Use FSDP for DIT model"}),
                "use_usp": ("BOOLEAN", {"default": False, "tooltip": "Use USP (Ultra-Scalable Parallelism)"}),
                "t5_cpu": ("BOOLEAN", {"default": False, "tooltip": "Load T5 on CPU"}),
            },
            "optional": {
                "param_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16", "tooltip": "Parameter dtype for FSDP"}),
                "reduce_dtype": (["float32", "bfloat16", "float16"], {"default": "float32", "tooltip": "Reduce dtype for FSDP"}),
                "buffer_dtype": (["float32", "bfloat16", "float16"], {"default": "float32", "tooltip": "Buffer dtype for FSDP"}),
                "sharding_strategy": (["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"], {"default": "FULL_SHARD", "tooltip": "FSDP sharding strategy"}),
                "sync_module_states": ("BOOLEAN", {"default": True, "tooltip": "Sync module states in FSDP"}),
            }
        }

    RETURN_TYPES = ("WANDISTRIBUTEDCONFIG",)
    RETURN_NAMES = ("wan_distributed_config",)
    FUNCTION = "create_config"
    CATEGORY = "WanVideoWrapper/Distributed"
    DESCRIPTION = "Configure Wan2.1 distributed inference settings"

    def create_config(self, **kwargs):
        # Convert string dtypes to torch dtypes
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        
        if "param_dtype" in kwargs:
            kwargs["param_dtype"] = dtype_map[kwargs["param_dtype"]]
        if "reduce_dtype" in kwargs:
            kwargs["reduce_dtype"] = dtype_map[kwargs["reduce_dtype"]]
        if "buffer_dtype" in kwargs:
            kwargs["buffer_dtype"] = dtype_map[kwargs["buffer_dtype"]]
        
        # Convert sharding strategy string to enum
        if "sharding_strategy" in kwargs:
            strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD
            }
            kwargs["sharding_strategy"] = strategy_map[kwargs["sharding_strategy"]]
        
        config = WanDistributedConfig(**kwargs)
        return (config,)

class WanDistributedModelLoader:
    """ComfyUI node for loading WanVideo models with Wan2.1 distributed inference"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "WanVideo model to load"}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "wan_distributed_config": ("WANDISTRIBUTEDCONFIG", {"tooltip": "Wan2.1 distributed configuration"}),
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
                "lora": ("WANVIDLORA", {"default": None}),
                "vace_model": ("VACEPATH", {"default": None}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/Distributed"
    DESCRIPTION = "Load WanVideo model with Wan2.1 distributed inference support"

    def loadmodel(self, model, base_precision, wan_distributed_config, attention_mode="sdpa", lora=None, vace_model=None):
        log.info("=" * 80)
        log.info("🚀 STARTING WAN2.1 DISTRIBUTED MODEL LOADER")
        log.info("=" * 80)
        log.info(f"📋 Input parameters:")
        log.info(f"   - Model: {model}")
        log.info(f"   - Base precision: {base_precision}")
        log.info(f"   - Attention mode: {attention_mode}")
        log.info(f"   - World size: {wan_distributed_config.world_size}")
        log.info(f"   - Use FSDP: {wan_distributed_config.use_fsdp}")
        log.info(f"   - Use context parallel: {wan_distributed_config.use_context_parallel}")
        log.info(f"   - LoRA: {lora is not None}")
        log.info(f"   - VACE model: {vace_model is not None}")
        
        # Check FSDP availability
        log.info("🔍 Checking FSDP availability...")
        if not FSDP_AVAILABLE:
            log.error("❌ FSDP not available!")
            raise ImportError("FSDP is required for distributed inference. Please install PyTorch with distributed support.")
        log.info("✅ FSDP is available")
        
        # Check GPU availability
        log.info("🔍 Checking GPU availability...")
        available_gpus = torch.cuda.device_count()
        log.info(f"   - Available GPUs: {available_gpus}")
        log.info(f"   - Requested GPUs: {wan_distributed_config.world_size}")
        
        if available_gpus < wan_distributed_config.world_size:
            log.error(f"❌ Not enough GPUs! Requested {wan_distributed_config.world_size} but only {available_gpus} available")
            raise ValueError(f"Requested {wan_distributed_config.world_size} GPUs but only {available_gpus} are available")
        log.info("✅ GPU count is sufficient")
        
        log.info(f"🎯 Loading WanVideo model with Wan2.1 distributed inference on {wan_distributed_config.world_size} GPUs")
        
        # Unload existing models
        log.info("🧹 Unloading existing models...")
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        log.info("✅ Models unloaded and cache cleared")
        
        # Set up device and dtype
        log.info("⚙️ Setting up devices and data types...")
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]
        log.info(f"   - Main device: {device}")
        log.info(f"   - Offload device: {offload_device}")
        log.info(f"   - Base dtype: {base_dtype}")
        log.info("✅ Device setup complete")
        
        # Load model state dict
        log.info("📁 Loading model state dict...")
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        log.info(f"   - Model path: {model_path}")
        log.info(f"   - Loading to device: {offload_device}")
        
        try:
            sd = load_torch_file(model_path, device=offload_device, safe_load=True)
            log.info(f"✅ Model loaded successfully. State dict keys: {len(sd)}")
        except Exception as e:
            log.error(f"❌ Failed to load model: {e}")
            raise
        
        # Handle VACE model if provided
        if vace_model is not None:
            log.info("🔧 Loading VACE model...")
            try:
                vace_sd = load_torch_file(vace_model["path"], device=offload_device, safe_load=True)
                sd.update(vace_sd)
                log.info(f"✅ VACE model loaded. Total keys: {len(sd)}")
            except Exception as e:
                log.error(f"❌ Failed to load VACE model: {e}")
                raise
        else:
            log.info("ℹ️ No VACE model provided")
        
        # Standardize state dict keys
        log.info("🔧 Standardizing state dict keys...")
        first_key = next(iter(sd))
        log.info(f"   - First key: {first_key}")
        
        if first_key.startswith("model.diffusion_model."):
            log.info("   - Detected 'model.diffusion_model.' prefix, removing...")
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
            log.info("✅ Removed 'model.diffusion_model.' prefix")
        elif first_key.startswith("model."):
            log.info("   - Detected 'model.' prefix, removing...")
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
            log.info("✅ Removed 'model.' prefix")
        else:
            log.info("   - No prefix detected, keys are already standardized")
        
        log.info(f"   - Final key count: {len(sd)}")
        
        # Validate model
        log.info("🔍 Validating model structure...")
        if not "patch_embedding.weight" in sd:
            log.error("❌ Invalid WanVideo model: missing 'patch_embedding.weight'")
            raise ValueError("Invalid WanVideo model selected")
        log.info("✅ Model validation passed")
        
        # Extract model configuration
        log.info("📐 Extracting model configuration...")
        try:
            dim = sd["patch_embedding.weight"].shape[0]
            in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
            out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
            in_channels = sd["patch_embedding.weight"].shape[1]
            ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
            ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]
            
            log.info(f"   - Dimension: {dim}")
            log.info(f"   - In features: {in_features}")
            log.info(f"   - Out features: {out_features}")
            log.info(f"   - In channels: {in_channels}")
            log.info(f"   - FFN dim: {ffn_dim}")
            log.info(f"   - FFN2 dim: {ffn2_dim}")
            log.info("✅ Model configuration extracted")
        except Exception as e:
            log.error(f"❌ Failed to extract model configuration: {e}")
            raise
        
        # Determine model type
        log.info("🏷️ Determining model type...")
        if not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn"
            log.info("   - Model type: no_cross_attn (no text embedding)")
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
            log.info("   - Model type: fl2v (FLF2V model)")
        elif in_channels in [36, 48]:
            model_type = "i2v"
            log.info(f"   - Model type: i2v (in_channels: {in_channels})")
        elif in_channels == 16:
            model_type = "t2v"
            log.info("   - Model type: t2v (in_channels: 16)")
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"
            log.info("   - Model type: t2v (with control adapter)")
        else:
            model_type = "unknown"
            log.warning(f"   - Model type: unknown (in_channels: {in_channels})")
        
        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30
        log.info(f"   - Num heads: {num_heads}")
        log.info(f"   - Num layers: {num_layers}")
        log.info(f"✅ Model type determined: {model_type}")
        
        # Handle VACE layers
        log.info("🔧 Handling VACE layers...")
        vace_layers, vace_in_dim = None, None
        if "vace_blocks.0.after_proj.weight" in sd:
            log.info("   - VACE blocks detected in model")
            if in_channels != 16:
                log.error("❌ VACE only works properly with T2V models")
                raise ValueError("VACE only works properly with T2V models.")
            model_type = "t2v"
            if dim == 5120:
                vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
                log.info("   - Using 14B VACE layers: [0, 5, 10, 15, 20, 25, 30, 35]")
            else:
                vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
                log.info("   - Using 1.3B VACE layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]")
            vace_in_dim = 96
            log.info(f"   - VACE input dimension: {vace_in_dim}")
            log.info("✅ VACE layers configured")
        else:
            log.info("ℹ️ No VACE blocks detected")
        
        log.info(f"📊 Final model configuration:")
        log.info(f"   - Model type: {model_type}")
        log.info(f"   - Num heads: {num_heads}")
        log.info(f"   - Num layers: {num_layers}")
        log.info(f"   - VACE layers: {vace_layers}")
        log.info(f"   - VACE in dim: {vace_in_dim}")
        
        # Create transformer configuration
        log.info("⚙️ Creating transformer configuration...")
        transformer_config = {
            "dim": dim,
            "in_features": in_features,
            "out_features": out_features,
            "ffn_dim": ffn_dim,
            "ffn2_dim": ffn2_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "rope_func": "comfy",
            "main_device": device,
            "offload_device": offload_device,
            "vace_layers": vace_layers,
            "vace_in_dim": vace_in_dim,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
        }
        
        log.info(f"   - Attention mode: {attention_mode}")
        log.info(f"   - Inject sample info: {transformer_config['inject_sample_info']}")
        log.info(f"   - Add ref conv: {transformer_config['add_ref_conv']}")
        log.info(f"   - Add control adapter: {transformer_config['add_control_adapter']}")
        log.info("✅ Transformer configuration created")
        
        # Create WanVideo model
        log.info("🏗️ Creating WanVideo model with empty weights...")
        try:
            with init_empty_weights():
                transformer = WanModel(**transformer_config)
            transformer.eval()
            log.info("✅ WanVideo model created successfully")
        except Exception as e:
            log.error(f"❌ Failed to create WanVideo model: {e}")
            raise
        
        # Create ComfyUI model wrapper
        log.info("🎭 Creating ComfyUI model wrapper...")
        try:
            comfy_model = WanDistributedModel(
                WanVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )
            
            comfy_model.diffusion_model = transformer
            comfy_model.load_device = offload_device
            
            # Ensure the model maintains consistent dtype
            comfy_model.base_dtype = base_dtype
            log.info(f"   - Model base dtype: {base_dtype}")
            log.info("✅ ComfyUI model wrapper created")
        except Exception as e:
            log.error(f"❌ Failed to create ComfyUI model wrapper: {e}")
            raise
        
        # Load model weights
        log.info("⚖️ Loading model weights...")
        try:
            param_count = sum(1 for _ in transformer.named_parameters())
            log.info(f"   - Total parameters to load: {param_count}")
            log.info(f"   - Target dtype: {base_dtype}")
            pbar = ProgressBar(param_count)
            
            loaded_count = 0
            for name, param in tqdm(transformer.named_parameters(), 
                    desc=f"Loading transformer parameters", 
                    total=param_count,
                    leave=True):
                try:
                    # Ensure the loaded weight has the correct dtype
                    weight_value = sd[name].to(dtype=base_dtype)
                    set_module_tensor_to_device(transformer, name, device=offload_device, dtype=base_dtype, value=weight_value)
                    loaded_count += 1
                    if loaded_count % 100 == 0:
                        log.info(f"   - Loaded {loaded_count}/{param_count} parameters")
                except Exception as e:
                    log.error(f"❌ Failed to load parameter {name}: {e}")
                    raise
                pbar.update(1)
            
            log.info(f"✅ Successfully loaded {loaded_count} parameters")
            
            # Ensure the model is in the correct dtype
            log.info("🔧 Converting model to target dtype...")
            transformer = transformer.to(dtype=base_dtype)
            log.info(f"✅ Model converted to {base_dtype}")
            
        except Exception as e:
            log.error(f"❌ Failed to load model weights: {e}")
            raise
        
        # Setup distributed inference
        log.info("🌐 Setting up distributed inference...")
        try:
            comfy_model.setup_distributed_inference(wan_distributed_config)
            log.info("✅ Distributed inference setup complete")
        except Exception as e:
            log.error(f"❌ Failed to setup distributed inference: {e}")
            raise
        
        # Handle LoRA if provided
        if lora is not None:
            log.warning("⚠️ LoRA support in distributed inference is experimental")
            # TODO: Implement LoRA support for distributed inference
        else:
            log.info("ℹ️ No LoRA provided")
        
        # Create model patcher
        log.info("🔧 Creating model patcher...")
        try:
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
            patcher.model.is_patched = True
            log.info("✅ Model patcher created")
        except Exception as e:
            log.error(f"❌ Failed to create model patcher: {e}")
            raise
        
        # Set model metadata
        log.info("📝 Setting model metadata...")
        try:
            patcher.model["dtype"] = base_dtype
            patcher.model["base_path"] = model_path
            patcher.model["model_name"] = model
            patcher.model["manual_offloading"] = True
            patcher.model["quantization"] = "disabled"
            patcher.model["auto_cpu_offload"] = False
            patcher.model["control_lora"] = False
            patcher.model["distributed_inference"] = True
            patcher.model["distributed_config"] = wan_distributed_config
            log.info("✅ Model metadata set")
        except Exception as e:
            log.error(f"❌ Failed to set model metadata: {e}")
            raise
        
        # Clean up
        log.info("🧹 Cleaning up...")
        try:
            del sd
            gc.collect()
            mm.soft_empty_cache()
            log.info("✅ Cleanup complete")
        except Exception as e:
            log.warning(f"⚠️ Cleanup warning: {e}")
        
        log.info("=" * 80)
        log.info("🎉 SUCCESSFULLY LOADED WANVIDEO MODEL WITH WAN2.1 DISTRIBUTED INFERENCE")
        log.info("=" * 80)
        log.info(f"📊 Summary:")
        log.info(f"   - Model: {model}")
        log.info(f"   - Type: {model_type}")
        log.info(f"   - GPUs: {wan_distributed_config.world_size}")
        log.info(f"   - FSDP: {wan_distributed_config.use_fsdp}")
        log.info(f"   - Context Parallel: {wan_distributed_config.use_context_parallel}")
        log.info(f"   - Parameters loaded: {loaded_count}")
        log.info("=" * 80)
        
        return (patcher,)

# Register the new nodes
NODE_CLASS_MAPPINGS = {
    "WanDistributedConfig": WanDistributedConfigNode,
    "WanDistributedModelLoader": WanDistributedModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanDistributedConfig": "Wan2.1 Distributed Config",
    "WanDistributedModelLoader": "Wan2.1 Distributed Model Loader",
} 