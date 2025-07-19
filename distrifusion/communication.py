"""
Asynchronous Communication for DistriFusion
Handles patch boundary synchronization across multiple GPUs
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import comfy.model_management as mm
from utils import log


class AsyncPatchCommunicator:
    """
    Handles asynchronous communication between patches on different GPUs
    Implements DistriFusion communication strategy
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.communication_streams = {}
        self.boundary_buffers = {}
        self.pending_operations = {}
        
        # Initialize NCCL for multi-GPU communication
        self.init_distributed()
        self.setup_communication_streams()
        
    def init_distributed(self):
        """Initialize distributed communication backend"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size
            )
        log.info(f"Initialized distributed communication for rank {self.rank}/{self.world_size}")
    
    def setup_communication_streams(self):
        """Setup CUDA streams for asynchronous communication"""
        for i in range(self.world_size):
            if i != self.rank:
                self.communication_streams[i] = torch.cuda.Stream(device=self.device)
        log.info(f"Setup {len(self.communication_streams)} communication streams")
    
    def get_neighbors(self, patch_layout: Tuple[int, int], device_id: int) -> List[int]:
        """
        Get neighboring patch IDs for spatial communication
        
        Args:
            patch_layout: (rows, cols) layout of patches
            device_id: Current device/patch ID
            
        Returns:
            List of neighboring device IDs
        """
        rows, cols = patch_layout
        row = device_id // cols
        col = device_id % cols
        
        neighbors = []
        
        # Add neighboring patches (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbor_id = new_row * cols + new_col
                neighbors.append(neighbor_id)
        
        return neighbors
    
    def get_temporal_neighbors(self, device_id: int, num_devices: int) -> List[int]:
        """Get neighboring devices for temporal communication"""
        neighbors = []
        
        if device_id > 0:
            neighbors.append(device_id - 1)
        if device_id < num_devices - 1:
            neighbors.append(device_id + 1)
        
        return neighbors
    
    def exchange_boundaries_sync(self, 
                                boundaries: Dict[str, torch.Tensor],
                                patch_layout: Tuple[int, int],
                                step: int) -> Dict[str, torch.Tensor]:
        """
        Synchronous boundary exchange for first step and warmup
        """
        received_boundaries = {}
        neighbors = self.get_neighbors(patch_layout, self.rank)
        
        # Create requests for all send/receive operations
        send_requests = []
        recv_requests = []
        recv_buffers = {}
        
        for neighbor in neighbors:
            # Determine which boundary to send/receive
            direction = self._get_direction_to_neighbor(patch_layout, self.rank, neighbor)
            
            if direction in boundaries:
                # Send boundary to neighbor
                boundary_data = boundaries[direction].contiguous()
                send_req = dist.isend(boundary_data, dst=neighbor)
                send_requests.append(send_req)
                
                # Prepare buffer for receiving
                recv_buffer = torch.empty_like(boundary_data)
                recv_req = dist.irecv(recv_buffer, src=neighbor)
                recv_requests.append(recv_req)
                recv_buffers[neighbor] = recv_buffer
        
        # Wait for all communications to complete
        for req in send_requests + recv_requests:
            req.wait()
        
        # Store received boundaries
        for neighbor, buffer in recv_buffers.items():
            direction = self._get_direction_from_neighbor(patch_layout, self.rank, neighbor)
            received_boundaries[f"neighbor_{neighbor}_{direction}"] = buffer
        
        return received_boundaries
    
    def exchange_boundaries_async(self,
                                 boundaries: Dict[str, torch.Tensor],
                                 cached_boundaries: Optional[Dict[str, torch.Tensor]],
                                 patch_layout: Tuple[int, int],
                                 step: int) -> Dict[str, torch.Tensor]:
        """
        Asynchronous boundary exchange using cached activations from previous step
        Core DistriFusion mechanism
        """
        if cached_boundaries is not None and step > 0:
            # Use cached boundaries from previous step for immediate computation
            return cached_boundaries
        
        # For first step or when cache is unavailable, fall back to sync
        return self.exchange_boundaries_sync(boundaries, patch_layout, step)
    
    def start_async_boundary_update(self,
                                   boundaries: Dict[str, torch.Tensor],
                                   patch_layout: Tuple[int, int],
                                   step: int) -> threading.Thread:
        """
        Start asynchronous update of boundary cache for next step
        This allows computation to continue while communication happens in background
        """
        def async_update():
            try:
                # Exchange boundaries asynchronously
                neighbors = self.get_neighbors(patch_layout, self.rank)
                
                for neighbor in neighbors:
                    direction = self._get_direction_to_neighbor(patch_layout, self.rank, neighbor)
                    
                    if direction in boundaries:
                        # Use dedicated stream for this communication
                        stream = self.communication_streams.get(neighbor)
                        if stream is not None:
                            with torch.cuda.stream(stream):
                                boundary_data = boundaries[direction].contiguous()
                                
                                # Non-blocking send
                                dist.isend(boundary_data, dst=neighbor)
                                
                                # Prepare buffer for next step
                                recv_buffer = torch.empty_like(boundary_data)
                                dist.irecv(recv_buffer, src=neighbor)
                                
                                # Cache for next step
                                cache_key = f"neighbor_{neighbor}_{direction}_step_{step + 1}"
                                self.boundary_buffers[cache_key] = recv_buffer
                                
            except Exception as e:
                log.error(f"Async boundary update failed: {e}")
        
        # Start background thread
        thread = threading.Thread(target=async_update)
        thread.start()
        return thread
    
    def _get_direction_to_neighbor(self, patch_layout: Tuple[int, int], 
                                  from_id: int, to_id: int) -> str:
        """Determine direction from current patch to neighbor"""
        rows, cols = patch_layout
        from_row, from_col = from_id // cols, from_id % cols
        to_row, to_col = to_id // cols, to_id % cols
        
        if to_row < from_row:
            return "top"
        elif to_row > from_row:
            return "bottom"
        elif to_col < from_col:
            return "left"
        elif to_col > from_col:
            return "right"
        else:
            return "unknown"
    
    def _get_direction_from_neighbor(self, patch_layout: Tuple[int, int],
                                   to_id: int, from_id: int) -> str:
        """Determine direction from neighbor to current patch"""
        # Reverse the direction
        direction = self._get_direction_to_neighbor(patch_layout, to_id, from_id)
        direction_map = {
            "top": "bottom",
            "bottom": "top", 
            "left": "right",
            "right": "left"
        }
        return direction_map.get(direction, "unknown")
    
    def apply_boundary_conditions(self,
                                 patch: torch.Tensor,
                                 received_boundaries: Dict[str, torch.Tensor],
                                 metadata: Dict[str, Any],
                                 overlap_size: int) -> torch.Tensor:
        """
        Apply boundary conditions from neighboring patches
        """
        if not received_boundaries:
            return patch
        
        # Create a copy to modify
        result = patch.clone()
        
        # Apply received boundary information
        for boundary_key, boundary_data in received_boundaries.items():
            if "neighbor" in boundary_key:
                try:
                    # Parse the boundary key to understand position
                    parts = boundary_key.split("_")
                    direction = parts[-1]
                    
                    # Apply boundary based on direction
                    if direction == "top" and patch.shape[-2] > overlap_size:
                        # Blend with top boundary
                        weight = 0.5  # Simple blending, can be made more sophisticated
                        result[:, :, :, :overlap_size, :] = (
                            weight * result[:, :, :, :overlap_size, :] +
                            (1 - weight) * boundary_data.to(result.device)
                        )
                    elif direction == "bottom" and patch.shape[-2] > overlap_size:
                        # Blend with bottom boundary
                        weight = 0.5
                        result[:, :, :, -overlap_size:, :] = (
                            weight * result[:, :, :, -overlap_size:, :] +
                            (1 - weight) * boundary_data.to(result.device)
                        )
                    elif direction == "left" and patch.shape[-1] > overlap_size:
                        # Blend with left boundary
                        weight = 0.5
                        result[:, :, :, :, :overlap_size] = (
                            weight * result[:, :, :, :, :overlap_size] +
                            (1 - weight) * boundary_data.to(result.device)
                        )
                    elif direction == "right" and patch.shape[-1] > overlap_size:
                        # Blend with right boundary
                        weight = 0.5
                        result[:, :, :, :, -overlap_size:] = (
                            weight * result[:, :, :, :, -overlap_size:] +
                            (1 - weight) * boundary_data.to(result.device)
                        )
                        
                except Exception as e:
                    log.warning(f"Failed to apply boundary condition {boundary_key}: {e}")
        
        return result
    
    def synchronize_all(self):
        """Synchronize all pending communications"""
        if dist.is_initialized():
            dist.barrier()
        
        # Wait for all communication streams
        for stream in self.communication_streams.values():
            stream.synchronize()
    
    def cleanup(self):
        """Cleanup distributed resources"""
        self.synchronize_all()
        self.boundary_buffers.clear()
        self.pending_operations.clear()
        
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedManager:
    """
    High-level manager for distributed DistriFusion inference
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.communicator = AsyncPatchCommunicator(world_size, rank)
        
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
    
    def scatter_input(self, input_tensor: torch.Tensor, patch_manager) -> torch.Tensor:
        """Scatter input tensor to appropriate patch for this rank"""
        if self.is_main_process():
            # Main process has the full tensor, split it
            patch, metadata = patch_manager.split_video_tensor(input_tensor, self.rank)
            return patch
        else:
            # Other processes receive their patch
            # In practice, this would involve tensor communication
            # For now, return a placeholder
            return torch.empty(1)  # This needs proper implementation
    
    def gather_output(self, output_patch: torch.Tensor, patch_manager, metadata_list: List[Dict]) -> Optional[torch.Tensor]:
        """Gather output patches from all processes"""
        if self.is_main_process():
            # Collect patches from all processes and reconstruct
            all_patches = [output_patch]  # Start with own patch
            
            # In practice, would gather from other processes
            # For now, return the single patch
            return patch_manager.reconstruct_tensor(all_patches, metadata_list)
        else:
            # Non-main processes send their patches
            return None
    
    def cleanup(self):
        """Cleanup distributed resources"""
        self.communicator.cleanup() 