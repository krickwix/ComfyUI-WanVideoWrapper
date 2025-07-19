"""
Patch Manager for DistriFusion
Handles spatial and temporal patch splitting for video tensors
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import comfy.model_management as mm
from ..utils import log


class PatchSplitMode(Enum):
    SPATIAL_ONLY = "spatial"
    TEMPORAL_ONLY = "temporal" 
    SPATIOTEMPORAL = "spatiotemporal"


@dataclass
class PatchConfig:
    """Configuration for patch splitting"""
    num_devices: int = 2
    split_mode: PatchSplitMode = PatchSplitMode.SPATIAL_ONLY
    patch_overlap: int = 8  # Overlap size for boundary interaction
    temporal_chunk_size: Optional[int] = None  # For temporal splitting
    spatial_patches_per_dim: int = 2  # For spatial splitting
    sync_first_step: bool = True  # DistriFusion: sync only first step
    async_boundary_update: bool = True  # Use async communication for boundaries
    warmup_steps: int = 4  # Number of warmup steps with full sync


class PatchManager:
    """
    Manages patch splitting and reconstruction for video tensors
    Implements DistriFusion displaced patch parallelism
    """
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.device_map = {}
        self.current_step = 0
        self.boundary_cache = {}  # Cache boundary activations from previous steps
        self.setup_device_mapping()
        
    def setup_device_mapping(self):
        """Setup mapping of patches to devices"""
        available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if len(available_devices) < self.config.num_devices:
            raise ValueError(f"Requested {self.config.num_devices} devices but only {len(available_devices)} available")
        
        self.devices = available_devices[:self.config.num_devices]
        log.info(f"DistriFusion using devices: {self.devices}")
        
    def split_video_tensor(self, x: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Split video tensor into patches for specific device
        
        Args:
            x: Video tensor [B, C, F, H, W]
            device_id: Target device ID (0 to num_devices-1)
            
        Returns:
            patch: Tensor patch for this device
            metadata: Information needed for reconstruction
        """
        B, C, F, H, W = x.shape
        metadata = {
            'original_shape': (B, C, F, H, W),
            'device_id': device_id,
            'step': self.current_step
        }
        
        if self.config.split_mode == PatchSplitMode.SPATIAL_ONLY:
            return self._split_spatial(x, device_id, metadata)
        elif self.config.split_mode == PatchSplitMode.TEMPORAL_ONLY:
            return self._split_temporal(x, device_id, metadata)
        else:  # SPATIOTEMPORAL
            return self._split_spatiotemporal(x, device_id, metadata)
    
    def _split_spatial(self, x: torch.Tensor, device_id: int, metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Split tensor spatially into patches"""
        B, C, F, H, W = x.shape
        
        # Calculate spatial patch dimensions
        patches_h = int(math.sqrt(self.config.num_devices))
        patches_w = self.config.num_devices // patches_h
        
        patch_h = H // patches_h
        patch_w = W // patches_w
        
        # Calculate patch indices for this device
        patch_row = device_id // patches_w
        patch_col = device_id % patches_w
        
        # Add overlap for boundary interaction
        h_start = max(0, patch_row * patch_h - self.config.patch_overlap)
        h_end = min(H, (patch_row + 1) * patch_h + self.config.patch_overlap)
        w_start = max(0, patch_col * patch_w - self.config.patch_overlap)
        w_end = min(W, (patch_col + 1) * patch_w + self.config.patch_overlap)
        
        patch = x[:, :, :, h_start:h_end, w_start:w_end].contiguous()
        
        metadata.update({
            'patch_bounds': (h_start, h_end, w_start, w_end),
            'core_bounds': (patch_row * patch_h, (patch_row + 1) * patch_h,
                           patch_col * patch_w, (patch_col + 1) * patch_w),
            'patches_layout': (patches_h, patches_w)
        })
        
        return patch, metadata
    
    def _split_temporal(self, x: torch.Tensor, device_id: int, metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Split tensor temporally into chunks"""
        B, C, F, H, W = x.shape
        
        chunk_size = self.config.temporal_chunk_size or (F // self.config.num_devices)
        
        # Calculate temporal bounds with overlap
        t_start = max(0, device_id * chunk_size - self.config.patch_overlap)
        t_end = min(F, (device_id + 1) * chunk_size + self.config.patch_overlap)
        
        patch = x[:, :, t_start:t_end, :, :].contiguous()
        
        metadata.update({
            'temporal_bounds': (t_start, t_end),
            'core_temporal_bounds': (device_id * chunk_size, (device_id + 1) * chunk_size),
            'chunk_size': chunk_size
        })
        
        return patch, metadata
    
    def _split_spatiotemporal(self, x: torch.Tensor, device_id: int, metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Split tensor both spatially and temporally"""
        # First split temporally
        temp_patch, temp_meta = self._split_temporal(x, device_id // 2, metadata)
        
        # Then split spatially within temporal chunk
        spatial_id = device_id % 2
        patch, spatial_meta = self._split_spatial(temp_patch, spatial_id, temp_meta)
        
        metadata.update(temp_meta)
        metadata.update(spatial_meta)
        
        return patch, metadata
    
    def get_boundary_regions(self, patch: torch.Tensor, metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Extract boundary regions for communication with neighboring patches
        """
        boundaries = {}
        
        if self.config.split_mode == PatchSplitMode.SPATIAL_ONLY:
            overlap = self.config.patch_overlap
            B, C, F, H, W = patch.shape
            
            # Extract boundary regions
            if H > overlap * 2:
                boundaries['top'] = patch[:, :, :, :overlap, :].clone()
                boundaries['bottom'] = patch[:, :, :, -overlap:, :].clone()
            
            if W > overlap * 2:
                boundaries['left'] = patch[:, :, :, :, :overlap].clone()
                boundaries['right'] = patch[:, :, :, :, -overlap:].clone()
        
        elif self.config.split_mode == PatchSplitMode.TEMPORAL_ONLY:
            overlap = self.config.patch_overlap
            B, C, F, H, W = patch.shape
            
            if F > overlap * 2:
                boundaries['temporal_start'] = patch[:, :, :overlap, :, :].clone()
                boundaries['temporal_end'] = patch[:, :, -overlap:, :, :].clone()
        
        return boundaries
    
    def reconstruct_tensor(self, patches: List[torch.Tensor], metadata_list: List[Dict]) -> torch.Tensor:
        """
        Reconstruct full tensor from distributed patches
        """
        if not patches:
            raise ValueError("No patches provided for reconstruction")
        
        # Get original shape from metadata
        original_shape = metadata_list[0]['original_shape']
        B, C, F, H, W = original_shape
        
        # Create output tensor on appropriate device
        device = patches[0].device
        output = torch.zeros(original_shape, dtype=patches[0].dtype, device=device)
        
        if self.config.split_mode == PatchSplitMode.SPATIAL_ONLY:
            return self._reconstruct_spatial(output, patches, metadata_list)
        elif self.config.split_mode == PatchSplitMode.TEMPORAL_ONLY:
            return self._reconstruct_temporal(output, patches, metadata_list)
        else:  # SPATIOTEMPORAL
            return self._reconstruct_spatiotemporal(output, patches, metadata_list)
    
    def _reconstruct_spatial(self, output: torch.Tensor, patches: List[torch.Tensor], metadata_list: List[Dict]) -> torch.Tensor:
        """Reconstruct spatially split tensor"""
        for patch, meta in zip(patches, metadata_list):
            core_bounds = meta['core_bounds']
            h_start, h_end, w_start, w_end = core_bounds
            
            # Calculate where to place this patch in the core region
            patch_bounds = meta['patch_bounds']
            ph_start, ph_end, pw_start, pw_end = patch_bounds
            
            # Calculate offset within patch to extract core region
            core_h_offset = h_start - ph_start
            core_w_offset = w_start - pw_start
            core_h_size = h_end - h_start
            core_w_size = w_end - w_start
            
            # Extract core region from patch and place in output
            core_patch = patch[:, :, :, 
                             core_h_offset:core_h_offset + core_h_size,
                             core_w_offset:core_w_offset + core_w_size]
            
            output[:, :, :, h_start:h_end, w_start:w_end] = core_patch
        
        return output
    
    def _reconstruct_temporal(self, output: torch.Tensor, patches: List[torch.Tensor], metadata_list: List[Dict]) -> torch.Tensor:
        """Reconstruct temporally split tensor"""
        for patch, meta in zip(patches, metadata_list):
            core_bounds = meta['core_temporal_bounds']
            t_start, t_end = core_bounds
            
            # Calculate offset within patch
            temporal_bounds = meta['temporal_bounds']
            pt_start, pt_end = temporal_bounds
            
            core_t_offset = t_start - pt_start
            core_t_size = t_end - t_start
            
            # Extract core region and place in output
            core_patch = patch[:, :, core_t_offset:core_t_offset + core_t_size, :, :]
            output[:, :, t_start:t_end, :, :] = core_patch
        
        return output
    
    def _reconstruct_spatiotemporal(self, output: torch.Tensor, patches: List[torch.Tensor], metadata_list: List[Dict]) -> torch.Tensor:
        """Reconstruct spatiotemporally split tensor"""
        # Group patches by temporal chunk first, then reconstruct spatially within each chunk
        temporal_groups = {}
        
        for patch, meta in zip(patches, metadata_list):
            temporal_bounds = meta['core_temporal_bounds']
            t_key = temporal_bounds
            
            if t_key not in temporal_groups:
                temporal_groups[t_key] = []
            temporal_groups[t_key].append((patch, meta))
        
        # Reconstruct each temporal chunk spatially
        for t_bounds, chunk_data in temporal_groups.items():
            t_start, t_end = t_bounds
            chunk_patches, chunk_metadata = zip(*chunk_data)
            
            # Create temporary output for this chunk
            chunk_shape = (output.shape[0], output.shape[1], t_end - t_start, output.shape[3], output.shape[4])
            chunk_output = torch.zeros(chunk_shape, dtype=output.dtype, device=output.device)
            
            # Reconstruct spatially within chunk
            chunk_output = self._reconstruct_spatial(chunk_output, chunk_patches, chunk_metadata)
            
            # Place in final output
            output[:, :, t_start:t_end, :, :] = chunk_output
        
        return output
    
    def should_sync_step(self, step: int) -> bool:
        """Determine if this step should use synchronous communication"""
        if step == 0 and self.config.sync_first_step:
            return True
        if step < self.config.warmup_steps:
            return True
        return False
    
    def update_step(self, step: int):
        """Update current step for DistriFusion logic"""
        self.current_step = step
    
    def cache_boundaries(self, device_id: int, boundaries: Dict[str, torch.Tensor], step: int):
        """Cache boundary activations for asynchronous reuse"""
        cache_key = (device_id, step)
        self.boundary_cache[cache_key] = boundaries
    
    def get_cached_boundaries(self, device_id: int, step: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached boundary activations from previous step"""
        cache_key = (device_id, step - 1)  # Get from previous step
        return self.boundary_cache.get(cache_key) 