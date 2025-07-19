# DistriFusion: Distributed Parallel Inference for WanVideo Models
# Implements displaced patch parallelism for multi-GPU inference

from .distrifusion_wrapper import DistriFusionWanModel
from .patch_manager import PatchManager, PatchConfig
from .communication import AsyncPatchCommunicator
from .distributed_nodes import (
    DistriFusionWanVideoModelLoader,
    DistriFusionWanVideoSampler
)

__all__ = [
    'DistriFusionWanModel',
    'PatchManager', 
    'PatchConfig',
    'AsyncPatchCommunicator',
    'DistriFusionWanVideoModelLoader',
    'DistriFusionWanVideoSampler'
]

__version__ = "1.0.0" 