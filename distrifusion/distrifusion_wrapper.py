"""
DistriFusion Wrapper for WanVideo Models
Implements displaced patch parallelism for distributed inference
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import time
try:
    from ..wanvideo.modules.model import WanModel
except ImportError:
    # Fallback for different import contexts
    try:
        from wanvideo.modules.model import WanModel
    except ImportError:
        WanModel = None
        print("Warning: WanModel not available")

from .patch_manager import PatchManager, PatchConfig, PatchSplitMode
from .communication import AsyncPatchCommunicator, DistributedManager
import comfy.model_management as mm

try:
    from ..utils import log
except ImportError:
    # Fallback for different import contexts
    import logging
    log = logging.getLogger(__name__)


class DistriFusionWanModel(nn.Module):
    """
    DistriFusion wrapper for WanVideo models
    Implements displaced patch parallelism for multi-GPU inference
    """
    
    def __init__(self, 
                 wan_model,  # Remove type hint since WanModel might be None
                 patch_config: PatchConfig,
                 world_size: int = 2,
                 rank: int = 0):
        super().__init__()
        
        # Check if WanModel is available
        if WanModel is None:
            raise ImportError(
                "WanModel is not available. Please ensure that the wanvideo module "
                "is properly installed and accessible."
            )
        
        if wan_model is None:
            raise ValueError("wan_model cannot be None")
        
        if not isinstance(wan_model, WanModel):
            raise TypeError(f"Expected WanModel, got {type(wan_model)}")
        
        self.wan_model = wan_model
        self.patch_config = patch_config
        self.world_size = world_size
        self.rank = rank
        
        # Initialize distributed components
        self.patch_manager = PatchManager(patch_config)
        self.distributed_manager = DistributedManager(world_size, rank)
        
        # Track inference state
        self.current_step = 0
        self.patch_cache = {}
        self.boundary_cache = {}
        self.async_threads = []
        
        # Device management
        self.device = torch.device(f"cuda:{rank}")
        self.wan_model.to(self.device)
        
        log.info(f"DistriFusion initialized for rank {rank}/{world_size}")
        log.info(f"Patch config: {patch_config}")
    
    def forward(self, 
                x: Union[torch.Tensor, List[torch.Tensor]],
                t: torch.Tensor,
                context: List[torch.Tensor],
                **kwargs) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[Any]]:
        """
        Forward pass with DistriFusion patch parallelism
        
        Args:
            x: Input latent(s) - either single tensor or list of tensors
            t: Timestep tensor
            context: Text context embeddings
            **kwargs: Additional arguments for WanModel
            
        Returns:
            Output latents and prediction ID
        """
        # Handle both single tensor and list inputs
        if isinstance(x, torch.Tensor):
            x_list = [x]
            single_input = True
        else:
            x_list = x
            single_input = False
        
        # Process each tensor in the list
        processed_tensors = []
        pred_id = None
        
        for i, tensor in enumerate(x_list):
            processed_tensor, tensor_pred_id = self._process_single_tensor(
                tensor, t, context, **kwargs
            )
            processed_tensors.append(processed_tensor)
            if pred_id is None:
                pred_id = tensor_pred_id
        
        # Return in same format as input
        if single_input:
            return processed_tensors[0], pred_id
        else:
            return processed_tensors, pred_id
    
    def _process_single_tensor(self,
                              x: torch.Tensor,
                              t: torch.Tensor,
                              context: List[torch.Tensor],
                              **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Process a single tensor with DistriFusion
        """
        B, C, F, H, W = x.shape
        
        # Step 1: Split tensor into patches
        patch, metadata = self.patch_manager.split_video_tensor(x, self.rank)
        patch = patch.to(self.device)
        
        # Step 2: Handle boundary communication
        boundaries = self.patch_manager.get_boundary_regions(patch, metadata)
        received_boundaries = self._handle_boundary_communication(boundaries, metadata)
        
        # Step 3: Apply boundary conditions to patch
        if received_boundaries:
            patch = self.distributed_manager.communicator.apply_boundary_conditions(
                patch, received_boundaries, metadata, self.patch_config.patch_overlap
            )
        
        # Step 4: Process patch through WanModel
        # Adjust input dimensions for patch
        x_patch_list = [patch]
        
        # Process through original WanModel
        output_patch_list, pred_id = self.wan_model(
            x_patch_list, t, context, **kwargs
        )
        output_patch = output_patch_list[0] if isinstance(output_patch_list, list) else output_patch_list
        
        # Step 5: Cache boundaries for next step (asynchronous)
        if self.patch_config.async_boundary_update and self.current_step > 0:
            self._start_async_boundary_cache(output_patch, metadata)
        
        # Step 6: Gather and reconstruct if main process
        if self.distributed_manager.is_main_process():
            # For now, just return the patch (full gathering implementation needed)
            reconstructed = self._reconstruct_from_patches([output_patch], [metadata])
            return reconstructed, pred_id
        else:
            # Non-main processes return their patch
            return output_patch, pred_id
    
    def _handle_boundary_communication(self,
                                     boundaries: Dict[str, torch.Tensor],
                                     metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Handle communication of boundary regions between patches
        """
        step = self.current_step
        is_sync_step = self.patch_manager.should_sync_step(step)
        
        if is_sync_step:
            # Synchronous communication for first step and warmup
            log.debug(f"Step {step}: Using synchronous boundary communication")
            
            if self.patch_config.split_mode == PatchSplitMode.SPATIAL_ONLY:
                patch_layout = metadata.get('patches_layout', (2, 1))
                return self.distributed_manager.communicator.exchange_boundaries_sync(
                    boundaries, patch_layout, step
                )
            else:
                # Handle temporal communication
                return self._handle_temporal_communication_sync(boundaries, step)
        else:
            # Asynchronous communication using cached boundaries
            log.debug(f"Step {step}: Using asynchronous boundary communication")
            
            cached_boundaries = self.patch_manager.get_cached_boundaries(self.rank, step)
            
            if self.patch_config.split_mode == PatchSplitMode.SPATIAL_ONLY:
                patch_layout = metadata.get('patches_layout', (2, 1))
                return self.distributed_manager.communicator.exchange_boundaries_async(
                    boundaries, cached_boundaries, patch_layout, step
                )
            else:
                return self._handle_temporal_communication_async(boundaries, cached_boundaries, step)
    
    def _handle_temporal_communication_sync(self,
                                          boundaries: Dict[str, torch.Tensor],
                                          step: int) -> Dict[str, torch.Tensor]:
        """Handle synchronous temporal boundary communication"""
        received_boundaries = {}
        
        neighbors = self.distributed_manager.communicator.get_temporal_neighbors(
            self.rank, self.world_size
        )
        
        for neighbor in neighbors:
            if self.rank < neighbor and 'temporal_end' in boundaries:
                # Send end to next temporal chunk
                dist.send(boundaries['temporal_end'], dst=neighbor)
            elif self.rank > neighbor and 'temporal_start' in boundaries:
                # Receive start from previous temporal chunk
                recv_buffer = torch.empty_like(boundaries['temporal_start'])
                dist.recv(recv_buffer, src=neighbor)
                received_boundaries[f'neighbor_{neighbor}_temporal'] = recv_buffer
        
        return received_boundaries
    
    def _handle_temporal_communication_async(self,
                                           boundaries: Dict[str, torch.Tensor],
                                           cached_boundaries: Optional[Dict[str, torch.Tensor]],
                                           step: int) -> Dict[str, torch.Tensor]:
        """Handle asynchronous temporal boundary communication"""
        if cached_boundaries is not None:
            return cached_boundaries
        
        # Fall back to synchronous for first step
        return self._handle_temporal_communication_sync(boundaries, step)
    
    def _start_async_boundary_cache(self,
                                   output_patch: torch.Tensor,
                                   metadata: Dict[str, Any]):
        """Start asynchronous caching of boundaries for next step"""
        def cache_boundaries():
            try:
                # Extract boundaries from output
                boundaries = self.patch_manager.get_boundary_regions(output_patch, metadata)
                
                # Cache for next step
                next_step = self.current_step + 1
                self.patch_manager.cache_boundaries(self.rank, boundaries, next_step)
                
                # Start async boundary update
                if self.patch_config.split_mode == PatchSplitMode.SPATIAL_ONLY:
                    patch_layout = metadata.get('patches_layout', (2, 1))
                    thread = self.distributed_manager.communicator.start_async_boundary_update(
                        boundaries, patch_layout, self.current_step
                    )
                    self.async_threads.append(thread)
                
                log.debug(f"Cached boundaries for step {next_step}")
                
            except Exception as e:
                log.error(f"Failed to cache boundaries: {e}")
        
        # Start in background thread
        thread = threading.Thread(target=cache_boundaries)
        thread.start()
        self.async_threads.append(thread)
    
    def _reconstruct_from_patches(self,
                                 patches: List[torch.Tensor],
                                 metadata_list: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Reconstruct full tensor from patches
        Currently simplified - full implementation would gather from all processes
        """
        if len(patches) == 1 and self.world_size == 1:
            # Single GPU case
            return patches[0]
        
        # For multi-GPU, we need to gather all patches
        # This is a simplified version - full implementation needed
        return self.patch_manager.reconstruct_tensor(patches, metadata_list)
    
    def update_step(self, step: int):
        """Update current step for DistriFusion logic"""
        self.current_step = step
        self.patch_manager.update_step(step)
    
    def synchronize(self):
        """Synchronize all distributed operations"""
        # Wait for async threads
        for thread in self.async_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.async_threads.clear()
        
        # Synchronize communicator
        self.distributed_manager.communicator.synchronize_all()
    
    def cleanup(self):
        """Cleanup distributed resources"""
        self.synchronize()
        self.distributed_manager.cleanup()
    
    def to(self, device):
        """Override to method to handle device placement"""
        super().to(device)
        self.wan_model.to(device)
        self.device = device
        return self
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying WanModel"""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.wan_model, name)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict of underlying WanModel"""
        return self.wan_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict into underlying WanModel"""
        return self.wan_model.load_state_dict(state_dict, *args, **kwargs)


def create_distrifusion_model(wan_model,  # Remove type hint since WanModel might be None
                             num_devices: int = 2,
                             split_mode: str = "spatial",
                             patch_overlap: int = 8,
                             world_size: int = 2,
                             rank: int = 0):
    """
    Factory function to create DistriFusion model
    
    Args:
        wan_model: Original WanVideo model
        num_devices: Number of GPUs to use
        split_mode: How to split patches ("spatial", "temporal", "spatiotemporal")
        patch_overlap: Overlap size for boundary interaction
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        DistriFusion wrapped model
    """
    # Check if WanModel is available
    if WanModel is None:
        raise ImportError(
            "WanModel is not available. Please ensure that the wanvideo module "
            "is properly installed and accessible."
        )
    
    # Convert string to enum
    mode_map = {
        "spatial": PatchSplitMode.SPATIAL_ONLY,
        "temporal": PatchSplitMode.TEMPORAL_ONLY,
        "spatiotemporal": PatchSplitMode.SPATIOTEMPORAL
    }
    
    patch_config = PatchConfig(
        num_devices=num_devices,
        split_mode=mode_map.get(split_mode, PatchSplitMode.SPATIAL_ONLY),
        patch_overlap=patch_overlap
    )
    
    return DistriFusionWanModel(
        wan_model=wan_model,
        patch_config=patch_config,
        world_size=world_size,
        rank=rank
    ) 