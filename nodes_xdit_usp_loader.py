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
                 world_size=2,
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
from .nodes_model_loading import WanVideoModel

class WanDistributedModel(WanVideoModel):
    """WanVideo model wrapper for Wan2.1 distributed inference"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributed_config = None
        self.fsdp_model = None
        self.is_distributed = False



    def setup_distributed_inference(self, config: WanDistributedConfig):
        """Setup Wan2.1 distributed inference"""
        if not FSDP_AVAILABLE:
            raise ImportError("FSDP is required for distributed inference")
        
        self.distributed_config = config
        
        # Initialize distributed environment
        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                rank=config.rank,
                world_size=config.world_size
            )
            self.is_distributed = True
            log.info(f"Initialized distributed environment with {config.world_size} processes")
        
        # Setup context parallel if enabled
        if config.use_context_parallel and XFUSER_AVAILABLE:
            if config.ulysses_size > 1 or config.ring_size > 1:
                assert config.ulysses_size * config.ring_size == config.world_size, \
                    f"The number of ulysses_size and ring_size should be equal to the world size."
                
                init_distributed_environment(
                    rank=dist.get_rank(), 
                    world_size=dist.get_world_size()
                )
                
                initialize_model_parallel(
                    sequence_parallel_degree=dist.get_world_size(),
                    ring_degree=config.ring_size,
                    ulysses_degree=config.ulysses_size,
                )
                log.info(f"Initialized context parallel: ulysses_size={config.ulysses_size}, ring_size={config.ring_size}")
        
        # Apply FSDP if enabled
        if config.use_fsdp:
            self.fsdp_model = self._apply_fsdp(config)
            log.info("Applied FSDP to model")
        
        log.info(f"Setup complete: FSDP={config.use_fsdp}, Context Parallel={config.use_context_parallel}")

    def _apply_fsdp(self, config):
        """Apply FSDP to the model"""
        if not hasattr(self, 'diffusion_model') or self.diffusion_model is None:
            raise ValueError("Model must be loaded before applying FSDP")
        
        # Define auto wrap policy for transformer blocks
        def auto_wrap_policy(module):
            return module in self.diffusion_model.blocks
        
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
        
        return fsdp_model

    def forward_distributed(self, *args, **kwargs):
        """Forward pass using distributed inference"""
        if self.fsdp_model is not None:
            return self.fsdp_model(*args, **kwargs)
        elif hasattr(self, 'diffusion_model'):
            return self.diffusion_model(*args, **kwargs)
        else:
            raise ValueError("No model available for inference")

    def cleanup(self):
        """Cleanup distributed resources"""
        if self.fsdp_model is not None:
            # Free FSDP storage
            for m in self.fsdp_model.modules():
                if isinstance(m, FSDP):
                    from torch.distributed.utils import _free_storage
                    _free_storage(m._handle.flat_param.data)
            del self.fsdp_model
            self.fsdp_model = None
        
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            self.is_distributed = False
        
        gc.collect()
        torch.cuda.empty_cache()

class WanDistributedConfigNode:
    """ComfyUI node for configuring Wan2.1 distributed settings"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "world_size": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Number of processes/GPUs"}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1, "tooltip": "Process rank (0 to world_size-1)"}),
                "backend": (["nccl", "gloo"], {"default": "nccl", "tooltip": "Distributed backend"}),
                "use_fsdp": ("BOOLEAN", {"default": True, "tooltip": "Use FSDP for model sharding"}),
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
        if not FSDP_AVAILABLE:
            raise ImportError("FSDP is required for distributed inference. Please install PyTorch with distributed support.")
        
        # Check if we have enough GPUs
        available_gpus = torch.cuda.device_count()
        if available_gpus < wan_distributed_config.world_size:
            raise ValueError(f"Requested {wan_distributed_config.world_size} GPUs but only {available_gpus} are available")
        
        log.info(f"Loading WanVideo model with Wan2.1 distributed inference on {wan_distributed_config.world_size} GPUs")
        
        # Unload existing models
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        
        # Set up device and dtype
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]
        
        # Load model state dict
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        
        # Handle VACE model if provided
        if vace_model is not None:
            vace_sd = load_torch_file(vace_model["path"], device=offload_device, safe_load=True)
            sd.update(vace_sd)
        
        # Standardize state dict keys
        first_key = next(iter(sd))
        if first_key.startswith("model.diffusion_model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        elif first_key.startswith("model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        
        # Validate model
        if not "patch_embedding.weight" in sd:
            raise ValueError("Invalid WanVideo model selected")
        
        # Extract model configuration
        dim = sd["patch_embedding.weight"].shape[0]
        in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
        ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]
        
        # Determine model type
        if not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn"
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
        elif in_channels in [36, 48]:
            model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"
        
        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30
        
        # Handle VACE layers
        vace_layers, vace_in_dim = None, None
        if "vace_blocks.0.after_proj.weight" in sd:
            if in_channels != 16:
                raise ValueError("VACE only works properly with T2V models.")
            model_type = "t2v"
            if dim == 5120:
                vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
            else:
                vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            vace_in_dim = 96
        
        log.info(f"Model type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")
        
        # Create transformer configuration
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
        
        # Create WanVideo model
        with init_empty_weights():
            transformer = WanModel(**transformer_config)
        transformer.eval()
        
        # Create ComfyUI model wrapper
        comfy_model = WanDistributedModel(
            comfy.model_base.ModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        
        comfy_model.diffusion_model = transformer
        comfy_model.load_device = offload_device
        
        # Load model weights
        log.info("Loading model weights...")
        param_count = sum(1 for _ in transformer.named_parameters())
        pbar = ProgressBar(param_count)
        
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters", 
                total=param_count,
                leave=True):
            set_module_tensor_to_device(transformer, name, device=offload_device, dtype=base_dtype, value=sd[name])
            pbar.update(1)
        
        # Setup distributed inference
        comfy_model.setup_distributed_inference(wan_distributed_config)
        
        # Handle LoRA if provided
        if lora is not None:
            log.warning("LoRA support in distributed inference is experimental")
            # TODO: Implement LoRA support for distributed inference
        
        # Create model patcher
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = True
        
        # Set model metadata
        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = True
        patcher.model["quantization"] = "disabled"
        patcher.model["auto_cpu_offload"] = False
        patcher.model["control_lora"] = False
        patcher.model["distributed_inference"] = True
        patcher.model["distributed_config"] = wan_distributed_config
        
        # Clean up
        del sd
        gc.collect()
        mm.soft_empty_cache()
        
        log.info("Successfully loaded WanVideo model with Wan2.1 distributed inference")
        
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