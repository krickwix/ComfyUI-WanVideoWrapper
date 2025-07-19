#!/usr/bin/env python3
"""
Official DistriFusion Integration for ComfyUI WanVideoWrapper
Uses the MIT-HAN-Lab DistriFusion implementation from:
https://github.com/mit-han-lab/distrifuser
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
import comfy.model_management as mm

try:
    from distrifuser import DistriFusion
    from distrifuser.utils import get_patch_size, get_patch_overlap
    DISTRIFUSER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Official DistriFusion not available: {e}")
    print("   Install with: pip install git+https://github.com/mit-han-lab/distrifuser.git")
    DISTRIFUSER_AVAILABLE = False
    DistriFusion = None

try:
    from utils import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class OfficialDistriFusionWanModel(nn.Module):
    """
    Wrapper for official DistriFusion with WanVideo models
    Uses the MIT-HAN-Lab implementation for proper distributed inference
    """
    
    def __init__(self, 
                 wan_model,
                 num_devices: int = 2,
                 patch_size: int = 64,
                 patch_overlap: int = 8,
                 split_mode: str = "spatial",
                 world_size: int = 2,
                 rank: int = 0):
        super().__init__()
        
        if not DISTRIFUSER_AVAILABLE:
            raise ImportError(
                "Official DistriFusion not available. Install with:\n"
                "pip install git+https://github.com/mit-han-lab/distrifuser.git"
            )
        
        self.wan_model = wan_model
        self.num_devices = num_devices
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.split_mode = split_mode
        self.world_size = world_size
        self.rank = rank
        
        # Initialize official DistriFusion
        self.distrifusion = DistriFusion(
            model=wan_model,
            num_devices=num_devices,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            split_mode=split_mode,
            world_size=world_size,
            rank=rank
        )
        
        # Track inference state
        self.current_step = 0
        
    def forward(self, 
                x: Union[torch.Tensor, List[torch.Tensor]],
                t: torch.Tensor,
                context: List[torch.Tensor],
                **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[Any]]:
        """
        Forward pass using official DistriFusion
        """
        # Update step tracking
        self.current_step += 1
        
        # Use official DistriFusion forward pass
        return self.distrifusion.forward(x, t, context, **kwargs)
    
    def update_step(self, step: int):
        """Update current inference step"""
        self.current_step = step
        if hasattr(self.distrifusion, 'update_step'):
            self.distrifusion.update_step(step)
    
    def synchronize(self):
        """Synchronize distributed processes"""
        if hasattr(self.distrifusion, 'synchronize'):
            self.distrifusion.synchronize()
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if hasattr(self.distrifusion, 'cleanup'):
            self.distrifusion.cleanup()
    
    def to(self, device):
        """Move model to device"""
        self.wan_model = self.wan_model.to(device)
        if hasattr(self.distrifusion, 'to'):
            self.distrifusion = self.distrifusion.to(device)
        return self
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wan_model, name)
    
    def state_dict(self, *args, **kwargs):
        """Get model state dict"""
        return self.wan_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load model state dict"""
        return self.wan_model.load_state_dict(state_dict, *args, **kwargs)


def create_official_distrifusion_model(wan_model,
                                      num_devices: int = 2,
                                      patch_size: int = 64,
                                      patch_overlap: int = 8,
                                      split_mode: str = "spatial",
                                      world_size: int = 2,
                                      rank: int = 0):
    """
    Factory function to create official DistriFusion model
    
    Args:
        wan_model: Original WanVideo model
        num_devices: Number of GPUs to use
        patch_size: Size of patches for splitting
        patch_overlap: Overlap size for boundary interaction
        split_mode: How to split patches ("spatial", "temporal", "spatiotemporal")
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        Official DistriFusion wrapped model
    """
    if not DISTRIFUSER_AVAILABLE:
        raise ImportError(
            "Official DistriFusion not available. Install with:\n"
            "pip install git+https://github.com/mit-han-lab/distrifuser.git"
        )
    
    return OfficialDistriFusionWanModel(
        wan_model=wan_model,
        num_devices=num_devices,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        split_mode=split_mode,
        world_size=world_size,
        rank=rank
    )


def get_official_distrifusion_config(num_devices: int = 2,
                                    patch_size: int = 64,
                                    patch_overlap: int = 8,
                                    split_mode: str = "spatial"):
    """
    Get configuration for official DistriFusion
    
    Args:
        num_devices: Number of GPUs
        patch_size: Patch size for splitting
        patch_overlap: Overlap size
        split_mode: Split mode
        
    Returns:
        Configuration dictionary
    """
    if not DISTRIFUSER_AVAILABLE:
        return None
    
    return {
        "num_devices": num_devices,
        "patch_size": patch_size,
        "patch_overlap": patch_overlap,
        "split_mode": split_mode,
        "library": "official_distrifuser",
        "version": "mit-han-lab"
    }


def check_official_distrifusion_availability():
    """
    Check if official DistriFusion is available
    
    Returns:
        Tuple of (available, error_message)
    """
    if not DISTRIFUSER_AVAILABLE:
        return False, "Official DistriFusion not installed. Install with: pip install git+https://github.com/mit-han-lab/distrifuser.git"
    
    try:
        # Test basic functionality
        from distrifuser import DistriFusion
        return True, "Official DistriFusion available"
    except Exception as e:
        return False, f"Official DistriFusion import error: {e}"


# Export main functions
__all__ = [
    'OfficialDistriFusionWanModel',
    'create_official_distrifusion_model',
    'get_official_distrifusion_config',
    'check_official_distrifusion_availability',
    'DISTRIFUSER_AVAILABLE'
] 