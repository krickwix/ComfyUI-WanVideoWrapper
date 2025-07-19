# DistriFusion: Distributed Parallel Inference for WanVideo Models
# Implements displaced patch parallelism for multi-GPU inference

from .distrifusion_wrapper import DistriFusionWanModel, create_distrifusion_model
from .patch_manager import PatchManager, PatchConfig, PatchSplitMode
from .communication import AsyncPatchCommunicator, DistributedManager
from .distributed_nodes import (
    DistriFusionWanVideoModelLoader,
    DistriFusionWanVideoSampler,
    DistriFusionSetup,
    DistriFusionStatus
)
from .distributed_model_loader import (
    DistriFusionModelLoader,
    DistriFusionDistributionConfig
)

__all__ = [
    'DistriFusionWanModel',
    'create_distrifusion_model',
    'PatchManager', 
    'PatchConfig',
    'PatchSplitMode',
    'AsyncPatchCommunicator',
    'DistributedManager',
    'DistriFusionWanVideoModelLoader',
    'DistriFusionWanVideoSampler',
    'DistriFusionSetup',
    'DistriFusionStatus',
    'DistriFusionModelLoader',
    'DistriFusionDistributionConfig'
]

__version__ = "1.0.0" 