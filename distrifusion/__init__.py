# DistriFusion: Distributed Parallel Inference for WanVideo Models
# Implements displaced patch parallelism for multi-GPU inference

__version__ = "1.0.0"
__all__ = []

# Import core components
try:
    from .distrifusion_wrapper import DistriFusionWanModel, create_distrifusion_model
    __all__.extend(['DistriFusionWanModel', 'create_distrifusion_model'])
except ImportError as e:
    print(f"Warning: Could not import DistriFusion wrapper: {e}")

try:
    from .patch_manager import PatchManager, PatchConfig, PatchSplitMode
    __all__.extend(['PatchManager', 'PatchConfig', 'PatchSplitMode'])
except ImportError as e:
    print(f"Warning: Could not import patch manager: {e}")

try:
    from .communication import AsyncPatchCommunicator, DistributedManager
    __all__.extend(['AsyncPatchCommunicator', 'DistributedManager'])
except ImportError as e:
    print(f"Warning: Could not import communication modules: {e}")

# Import node classes
try:
    from .distributed_nodes import (
        DistriFusionWanVideoModelLoader,
        DistriFusionWanVideoSampler,
        DistriFusionSetup,
        DistriFusionStatus
    )
    __all__.extend([
        'DistriFusionWanVideoModelLoader',
        'DistriFusionWanVideoSampler', 
        'DistriFusionSetup',
        'DistriFusionStatus'
    ])
except ImportError as e:
    print(f"Warning: Could not import distributed nodes: {e}")

try:
    from .distributed_model_loader import (
        DistriFusionModelLoader,
        DistriFusionDistributionConfig
    )
    __all__.extend([
        'DistriFusionModelLoader',
        'DistriFusionDistributionConfig'
    ])
except ImportError as e:
    print(f"Warning: Could not import model loader: {e}")

print(f"DistriFusion v{__version__} - Loaded {len(__all__)} components") 