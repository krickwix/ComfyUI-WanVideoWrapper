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

# Import xdit and USP dependencies
try:
    import xdit
    from xdit import XDitModel
    from xdit.config import XDitConfig
    from xdit.utils import get_xdit_config
    XDIT_AVAILABLE = True
except ImportError:
    XDIT_AVAILABLE = False
    log.warning("xdit not available. Please install xdit for distributed inference support.")

try:
    import usp
    from usp import USPDistributedInference
    USP_AVAILABLE = True
except ImportError:
    USP_AVAILABLE = False
    log.warning("usp not available. Please install usp for distributed inference support.")

class XDitUSPConfig:
    """Configuration class for xdit+USP distributed inference settings"""
    def __init__(self, 
                 num_gpus=2,
                 gpu_memory_fraction=0.9,
                 pipeline_parallel_size=1,
                 tensor_parallel_size=1,
                 data_parallel_size=1,
                 use_fp16=True,
                 use_bf16=False,
                 enable_activation_checkpointing=True,
                 enable_gradient_checkpointing=False,
                 max_batch_size=1,
                 max_sequence_length=2048,
                 overlap_p2p_comm=True,
                 use_flash_attention=True,
                 use_sdpa=True):
        self.num_gpus = num_gpus
        self.gpu_memory_fraction = gpu_memory_fraction
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.enable_activation_checkpointing = enable_activation_checkpointing
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.overlap_p2p_comm = overlap_p2p_comm
        self.use_flash_attention = use_flash_attention
        self.use_sdpa = use_sdpa

class XDitUSPWanVideoModel(comfy.model_base.BaseModel):
    """WanVideo model wrapper for xdit+USP distributed inference"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.xdit_model = None
        self.usp_inference = None
        self.distributed_config = None

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

    def setup_distributed_inference(self, config: XDitUSPConfig):
        """Setup xdit+USP distributed inference"""
        if not XDIT_AVAILABLE:
            raise ImportError("xdit is required for distributed inference")
        if not USP_AVAILABLE:
            raise ImportError("usp is required for distributed inference")
        
        self.distributed_config = config
        
        # Initialize xdit model
        xdit_config = XDitConfig(
            num_layers=self.diffusion_model.num_layers,
            hidden_size=self.diffusion_model.dim,
            num_attention_heads=self.diffusion_model.num_heads,
            intermediate_size=self.diffusion_model.ffn_dim,
            max_position_embeddings=config.max_sequence_length,
            use_flash_attention=config.use_flash_attention,
            use_sdpa=config.use_sdpa,
            pipeline_parallel_size=config.pipeline_parallel_size,
            tensor_parallel_size=config.tensor_parallel_size,
            data_parallel_size=config.data_parallel_size,
            enable_activation_checkpointing=config.enable_activation_checkpointing,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            overlap_p2p_comm=config.overlap_p2p_comm
        )
        
        self.xdit_model = XDitModel(xdit_config)
        
        # Initialize USP distributed inference
        self.usp_inference = USPDistributedInference(
            model=self.xdit_model,
            num_gpus=config.num_gpus,
            gpu_memory_fraction=config.gpu_memory_fraction,
            max_batch_size=config.max_batch_size,
            use_fp16=config.use_fp16,
            use_bf16=config.use_bf16
        )
        
        log.info(f"Initialized xdit+USP distributed inference with {config.num_gpus} GPUs")

    def convert_wan_to_xdit(self):
        """Convert WanVideo model weights to xdit format"""
        if self.xdit_model is None:
            raise ValueError("xdit model not initialized")
        
        # Convert WanVideo model weights to xdit format
        wan_state_dict = self.diffusion_model.state_dict()
        xdit_state_dict = {}
        
        # Mapping from WanVideo to xdit parameter names
        param_mapping = {
            'patch_embedding.weight': 'embeddings.patch_embedding.weight',
            'patch_embedding.bias': 'embeddings.patch_embedding.bias',
            'time_embedding.0.weight': 'embeddings.time_embedding.weight',
            'time_embedding.0.bias': 'embeddings.time_embedding.bias',
            'time_embedding.2.weight': 'embeddings.time_projection.weight',
            'time_embedding.2.bias': 'embeddings.time_projection.bias',
        }
        
        # Add transformer blocks mapping
        for i in range(self.diffusion_model.num_layers):
            block_prefix = f'blocks.{i}.'
            xdit_block_prefix = f'encoder.layers.{i}.'
            
            # Self attention
            param_mapping.update({
                f'{block_prefix}self_attn.q.weight': f'{xdit_block_prefix}self_attn.q_proj.weight',
                f'{block_prefix}self_attn.k.weight': f'{xdit_block_prefix}self_attn.k_proj.weight',
                f'{block_prefix}self_attn.v.weight': f'{xdit_block_prefix}self_attn.v_proj.weight',
                f'{block_prefix}self_attn.out.weight': f'{xdit_block_prefix}self_attn.out_proj.weight',
                f'{block_prefix}self_attn.q.bias': f'{xdit_block_prefix}self_attn.q_proj.bias',
                f'{block_prefix}self_attn.k.bias': f'{xdit_block_prefix}self_attn.k_proj.bias',
                f'{block_prefix}self_attn.v.bias': f'{xdit_block_prefix}self_attn.v_proj.bias',
                f'{block_prefix}self_attn.out.bias': f'{xdit_block_prefix}self_attn.out_proj.bias',
            })
            
            # Cross attention (if exists)
            if hasattr(self.diffusion_model.blocks[i], 'cross_attn'):
                param_mapping.update({
                    f'{block_prefix}cross_attn.q.weight': f'{xdit_block_prefix}cross_attn.q_proj.weight',
                    f'{block_prefix}cross_attn.k.weight': f'{xdit_block_prefix}cross_attn.k_proj.weight',
                    f'{block_prefix}cross_attn.v.weight': f'{xdit_block_prefix}cross_attn.v_proj.weight',
                    f'{block_prefix}cross_attn.out.weight': f'{xdit_block_prefix}cross_attn.out_proj.weight',
                    f'{block_prefix}cross_attn.q.bias': f'{xdit_block_prefix}cross_attn.q_proj.bias',
                    f'{block_prefix}cross_attn.k.bias': f'{xdit_block_prefix}cross_attn.k_proj.bias',
                    f'{block_prefix}cross_attn.v.bias': f'{xdit_block_prefix}cross_attn.v_proj.bias',
                    f'{block_prefix}cross_attn.out.bias': f'{xdit_block_prefix}cross_attn.out_proj.bias',
                })
            
            # FFN
            param_mapping.update({
                f'{block_prefix}ffn.0.weight': f'{xdit_block_prefix}mlp.fc1.weight',
                f'{block_prefix}ffn.0.bias': f'{xdit_block_prefix}mlp.fc1.bias',
                f'{block_prefix}ffn.2.weight': f'{xdit_block_prefix}mlp.fc2.weight',
                f'{block_prefix}ffn.2.bias': f'{xdit_block_prefix}mlp.fc2.bias',
            })
            
            # Layer norms
            param_mapping.update({
                f'{block_prefix}norm1.weight': f'{xdit_block_prefix}self_attn_layer_norm.weight',
                f'{block_prefix}norm1.bias': f'{xdit_block_prefix}self_attn_layer_norm.bias',
                f'{block_prefix}norm2.weight': f'{xdit_block_prefix}final_layer_norm.weight',
                f'{block_prefix}norm2.bias': f'{xdit_block_prefix}final_layer_norm.bias',
            })
        
        # Convert parameters
        for wan_key, xdit_key in param_mapping.items():
            if wan_key in wan_state_dict:
                xdit_state_dict[xdit_key] = wan_state_dict[wan_key]
        
        # Load converted weights into xdit model
        missing_keys, unexpected_keys = self.xdit_model.load_state_dict(xdit_state_dict, strict=False)
        
        if missing_keys:
            log.warning(f"Missing keys when converting to xdit: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys when converting to xdit: {unexpected_keys}")
        
        log.info("Successfully converted WanVideo model to xdit format")

    def forward_distributed(self, *args, **kwargs):
        """Forward pass using distributed inference"""
        if self.usp_inference is None:
            raise ValueError("USP inference not initialized")
        
        return self.usp_inference.forward(*args, **kwargs)

class XDitUSPConfigNode:
    """ComfyUI node for configuring xdit+USP settings"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Number of GPUs to use for distributed inference"}),
                "gpu_memory_fraction": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Fraction of GPU memory to use per GPU"}),
                "pipeline_parallel_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Pipeline parallel size"}),
                "tensor_parallel_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Tensor parallel size"}),
                "data_parallel_size": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Data parallel size"}),
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use FP16 precision for distributed inference"}),
                "use_bf16": ("BOOLEAN", {"default": False, "tooltip": "Use BF16 precision for distributed inference"}),
                "enable_activation_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "Enable activation checkpointing to save memory"}),
                "enable_gradient_checkpointing": ("BOOLEAN", {"default": False, "tooltip": "Enable gradient checkpointing"}),
                "max_batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Maximum batch size for distributed inference"}),
                "max_sequence_length": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512, "tooltip": "Maximum sequence length"}),
                "overlap_p2p_comm": ("BOOLEAN", {"default": True, "tooltip": "Overlap peer-to-peer communication with computation"}),
                "use_flash_attention": ("BOOLEAN", {"default": True, "tooltip": "Use Flash Attention for faster attention computation"}),
                "use_sdpa": ("BOOLEAN", {"default": True, "tooltip": "Use SDPA (Scaled Dot-Product Attention) for attention computation"}),
            }
        }

    RETURN_TYPES = ("XDITUSPCONFIG",)
    RETURN_NAMES = ("xdit_usp_config",)
    FUNCTION = "create_config"
    CATEGORY = "WanVideoWrapper/XDitUSP"
    DESCRIPTION = "Configure xdit+USP distributed inference settings"

    def create_config(self, **kwargs):
        config = XDitUSPConfig(**kwargs)
        return (config,)

class XDitUSPWanVideoModelLoader:
    """ComfyUI node for loading WanVideo models with xdit+USP distributed inference"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "WanVideo model to load"}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "xdit_usp_config": ("XDITUSPCONFIG", {"tooltip": "xdit+USP configuration"}),
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

    RETURN_TYPES = ("XDITUSPWANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/XDitUSP"
    DESCRIPTION = "Load WanVideo model with xdit+USP distributed inference support"

    def loadmodel(self, model, base_precision, xdit_usp_config, attention_mode="sdpa", lora=None, vace_model=None):
        if not XDIT_AVAILABLE:
            raise ImportError("xdit is required for distributed inference. Please install xdit.")
        if not USP_AVAILABLE:
            raise ImportError("usp is required for distributed inference. Please install usp.")
        
        # Check if we have enough GPUs
        available_gpus = torch.cuda.device_count()
        if available_gpus < xdit_usp_config.num_gpus:
            raise ValueError(f"Requested {xdit_usp_config.num_gpus} GPUs but only {available_gpus} are available")
        
        log.info(f"Loading WanVideo model with xdit+USP distributed inference on {xdit_usp_config.num_gpus} GPUs")
        
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
        comfy_model = XDitUSPWanVideoModel(
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
        comfy_model.setup_distributed_inference(xdit_usp_config)
        
        # Convert to xdit format
        comfy_model.convert_wan_to_xdit()
        
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
        
        # Clean up
        del sd
        gc.collect()
        mm.soft_empty_cache()
        
        log.info("Successfully loaded WanVideo model with xdit+USP distributed inference")
        
        return (patcher,)

# Register the new nodes
NODE_CLASS_MAPPINGS = {
    "XDitUSPConfig": XDitUSPConfigNode,
    "XDitUSPWanVideoModelLoader": XDitUSPWanVideoModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XDitUSPConfig": "XDit+USP Config",
    "XDitUSPWanVideoModelLoader": "XDit+USP WanVideo Model Loader",
} 