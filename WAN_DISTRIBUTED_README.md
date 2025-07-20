# Wan2.1 Distributed Inference for ComfyUI

This extension adds support for multi-GPU distributed inference using the official Wan2.1 distributed components (FSDP, xdit, USP) in ComfyUI.

## Overview

The Wan2.1 distributed inference integration leverages the official distributed components from the Wan2.1 repository, providing robust and optimized multi-GPU inference capabilities.

### Key Features

- **FSDP (Fully Sharded Data Parallel)**: Model sharding across GPUs
- **Context Parallel**: Sequence parallelism using xfuser
- **USP (Ultra-Scalable Parallelism)**: Advanced distributed strategies
- **Mixed Precision**: Support for FP16, BF16, and FP32
- **Memory Optimization**: Automatic memory management
- **Compatibility**: Uses official Wan2.1 distributed components

## Installation

### Prerequisites

- CUDA 11.8+ or ROCm 5.0+
- PyTorch 2.0+ with distributed support
- Multiple GPUs (2-8 recommended)
- Sufficient system memory (32GB+ recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch>=2.0.0 xfuser>=0.1.0
```

## Usage

### 1. Configure Wan2.1 Distributed Settings

Use the **Wan2.1 Distributed Config** node to configure distributed inference settings:

- **world_size**: Number of processes/GPUs (1-8)
- **rank**: Process rank (0 to world_size-1)
- **backend**: Distributed backend (nccl, gloo)
- **use_fsdp**: Use FSDP for model sharding
- **use_context_parallel**: Use context parallel (requires xfuser)
- **ulysses_size**: Ulysses size for context parallel (1-4)
- **ring_size**: Ring size for context parallel (1-4)
- **t5_fsdp**: Use FSDP for T5 text encoder
- **dit_fsdp**: Use FSDP for DIT model
- **use_usp**: Use USP (Ultra-Scalable Parallelism)
- **t5_cpu**: Load T5 on CPU

### 2. Load Model with Distributed Inference

Use the **Wan2.1 Distributed Model Loader** node:

1. Select your WanVideo model
2. Choose base precision (fp32, bf16, fp16)
3. Connect the Wan2.1 distributed config
4. Optionally configure attention mode and LoRA

### Example Workflow

```
Wan2.1 Distributed Config
├── world_size: 4
├── rank: 0
├── use_fsdp: True
├── use_context_parallel: False
└── param_dtype: bfloat16
    ↓
Wan2.1 Distributed Model Loader
├── model: wan2.1_14b.safetensors
├── base_precision: bf16
├── wan_distributed_config: [from config node]
└── attention_mode: sdpa
    ↓
[Your inference nodes]
```

## Configuration Guidelines

### GPU Memory Requirements

| Model Size | Single GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------------|------------|--------|--------|--------|
| 1.3B       | 8GB        | 4GB    | 2GB    | 1GB    |
| 14B        | 48GB       | 24GB   | 12GB   | 6GB    |

### Recommended Configurations

#### For 2 GPUs (24GB each)
```python
{
    "world_size": 2,
    "rank": 0,
    "use_fsdp": True,
    "use_context_parallel": False,
    "param_dtype": "bfloat16",
    "sharding_strategy": "FULL_SHARD"
}
```

#### For 4 GPUs (24GB each)
```python
{
    "world_size": 4,
    "rank": 0,
    "use_fsdp": True,
    "use_context_parallel": True,
    "ulysses_size": 2,
    "ring_size": 2,
    "param_dtype": "bfloat16"
}
```

#### For 8 GPUs (24GB each)
```python
{
    "world_size": 8,
    "rank": 0,
    "use_fsdp": True,
    "use_context_parallel": True,
    "ulysses_size": 4,
    "ring_size": 2,
    "param_dtype": "bfloat16"
}
```

## Performance Optimization

### Memory Optimization

1. **Use FSDP**: Enables model sharding across GPUs
2. **Mixed Precision**: Use BF16/FP16 for reduced memory usage
3. **Context Parallel**: Distribute sequence processing
4. **T5 CPU Offloading**: Load text encoder on CPU if needed

### Speed Optimization

1. **NCCL Backend**: Use NCCL for GPU communication
2. **Full Sharding**: Use FULL_SHARD strategy for maximum memory savings
3. **Context Parallel**: Enable for long sequences
4. **USP**: Enable for advanced parallelism

### Communication Optimization

1. **FSDP**: Efficient parameter sharding
2. **Context Parallel**: Sequence-level parallelism
3. **Ring Parallel**: Ring-based communication
4. **Ulysses Parallel**: Advanced attention parallelism

## Troubleshooting

### Common Issues

#### "Not enough GPUs available"
- Check `torch.cuda.device_count()` to verify available GPUs
- Reduce `world_size` in configuration

#### "Out of memory"
- Enable `use_fsdp`
- Use mixed precision (`param_dtype: "bfloat16"`)
- Enable context parallel for long sequences
- Use T5 CPU offloading

#### "ImportError: FSDP not available"
- Install PyTorch with distributed support
- Check CUDA compatibility

#### "ImportError: xfuser not available"
- Install xfuser: `pip install xfuser>=0.1.0`
- Verify PyTorch version compatibility

### Performance Debugging

1. **Monitor GPU Memory**: Use `nvidia-smi` to check memory usage
2. **Check Communication**: Monitor inter-GPU communication
3. **Profile Inference**: Use PyTorch profiler for bottlenecks
4. **Validate Configuration**: Ensure parallelism settings are valid

## Advanced Configuration

### FSDP Strategies

```python
# Full sharding (maximum memory savings)
{
    "sharding_strategy": "FULL_SHARD",
    "use_fsdp": True
}

# Gradient sharding only
{
    "sharding_strategy": "SHARD_GRAD_OP",
    "use_fsdp": True
}

# No sharding (for debugging)
{
    "sharding_strategy": "NO_SHARD",
    "use_fsdp": True
}
```

### Context Parallel Strategies

```python
# Sequence parallel only
{
    "use_context_parallel": True,
    "ulysses_size": 1,
    "ring_size": 4
}

# Ring parallel only
{
    "use_context_parallel": True,
    "ulysses_size": 4,
    "ring_size": 1
}

# Hybrid parallel
{
    "use_context_parallel": True,
    "ulysses_size": 2,
    "ring_size": 2
}
```

### Memory-Efficient Settings

```python
{
    "use_fsdp": True,
    "param_dtype": "bfloat16",
    "reduce_dtype": "float32",
    "buffer_dtype": "float32",
    "t5_cpu": True,
    "sync_module_states": True
}
```

### High-Performance Settings

```python
{
    "use_fsdp": True,
    "use_context_parallel": True,
    "ulysses_size": 2,
    "ring_size": 2,
    "param_dtype": "bfloat16",
    "use_usp": True,
    "backend": "nccl"
}
```

## Limitations

1. **LoRA Support**: Experimental, may not work with all LoRA types
2. **VACE Models**: Limited support for VACE in distributed mode
3. **Model Compatibility**: Only tested with WanVideo models
4. **Setup Complexity**: Requires proper GPU configuration
5. **xfuser Dependency**: Context parallel requires xfuser installation

## Future Enhancements

- [ ] Full LoRA support for distributed inference
- [ ] VACE model optimization
- [ ] Dynamic parallelism adjustment
- [ ] Automatic memory optimization
- [ ] Support for other model architectures
- [ ] Integration with other distributed frameworks

## Contributing

Contributions are welcome! Please:

1. Test with different GPU configurations
2. Report performance issues
3. Suggest optimization strategies
4. Add support for new features

## License

This extension follows the same license as the main ComfyUI-WanVideoWrapper project. 