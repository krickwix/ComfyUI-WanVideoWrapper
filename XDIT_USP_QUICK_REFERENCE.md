# XDit+USP Quick Reference

## Installation
```bash
# Automatic installation
./install_xdit_usp.sh

# Manual installation
pip install xdit>=0.1.0 usp>=0.1.0
```

## Quick Start

### 1. Configure XDit+USP Settings
- **Node**: `XDit+USP Config`
- **Key Settings**:
  - `num_gpus`: Number of GPUs (2-8)
  - `gpu_memory_fraction`: Memory per GPU (0.1-1.0)
  - `pipeline_parallel_size`: Pipeline parallelism (1-4)
  - `tensor_parallel_size`: Tensor parallelism (1-4)
  - `use_fp16`: Enable FP16 precision

### 2. Load Model
- **Node**: `XDit+USP WanVideo Model Loader`
- **Connect**: XDit+USP config from step 1
- **Select**: Your WanVideo model
- **Choose**: Base precision (bf16 recommended)

### 3. Complete Workflow
- Connect VAE and text encoder as usual
- Use standard WanVideo sampler
- Save video output

## Recommended Configurations

### 2 GPUs (24GB each)
```python
{
    "num_gpus": 2,
    "gpu_memory_fraction": 0.9,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 1,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

### 4 GPUs (24GB each)
```python
{
    "num_gpus": 4,
    "gpu_memory_fraction": 0.8,
    "pipeline_parallel_size": 2,
    "tensor_parallel_size": 2,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

### 8 GPUs (24GB each)
```python
{
    "num_gpus": 8,
    "gpu_memory_fraction": 0.7,
    "pipeline_parallel_size": 4,
    "tensor_parallel_size": 2,
    "use_fp16": True,
    "enable_activation_checkpointing": True
}
```

## Memory Requirements

| Model | Single GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|-------|------------|--------|--------|--------|
| 1.3B  | 8GB        | 4GB    | 2GB    | 1GB    |
| 14B   | 48GB       | 24GB   | 12GB   | 6GB    |

## Troubleshooting

### Common Issues
- **"Not enough GPUs"**: Reduce `num_gpus` or check `nvidia-smi`
- **"Out of memory"**: Reduce `gpu_memory_fraction` or enable activation checkpointing
- **"ImportError"**: Install dependencies with `./install_xdit_usp.sh`

### Performance Tips
- Use FP16 for better memory efficiency
- Enable activation checkpointing for large models
- Use Flash Attention when available
- Overlap peer-to-peer communication

## Node Reference

### XDit+USP Config
**Category**: `WanVideoWrapper/XDitUSP`

**Inputs**:
- `num_gpus` (INT): Number of GPUs (1-8)
- `gpu_memory_fraction` (FLOAT): Memory fraction per GPU (0.1-1.0)
- `pipeline_parallel_size` (INT): Pipeline parallel size (1-4)
- `tensor_parallel_size` (INT): Tensor parallel size (1-4)
- `data_parallel_size` (INT): Data parallel size (1-4)
- `use_fp16` (BOOLEAN): Enable FP16 precision
- `use_bf16` (BOOLEAN): Enable BF16 precision
- `enable_activation_checkpointing` (BOOLEAN): Memory optimization
- `enable_gradient_checkpointing` (BOOLEAN): Gradient optimization
- `max_batch_size` (INT): Maximum batch size (1-8)
- `max_sequence_length` (INT): Maximum sequence length (512-8192)
- `overlap_p2p_comm` (BOOLEAN): Overlap communication
- `use_flash_attention` (BOOLEAN): Use Flash Attention
- `use_sdpa` (BOOLEAN): Use SDPA

**Outputs**:
- `xdit_usp_config` (XDITUSPCONFIG): Configuration object

### XDit+USP WanVideo Model Loader
**Category**: `WanVideoWrapper/XDitUSP`

**Inputs**:
- `model` (STRING): WanVideo model filename
- `base_precision` (STRING): Precision (fp32, bf16, fp16)
- `xdit_usp_config` (XDITUSPCONFIG): Configuration from config node
- `attention_mode` (STRING): Attention mode (sdpa, flash_attn_2, etc.)
- `lora` (WANVIDLORA): LoRA model (experimental)
- `vace_model` (VACEPATH): VACE model path

**Outputs**:
- `model` (XDITUSPWANVIDEOMODEL): Loaded model with distributed inference

## Example Workflow
Load `example_workflows/xdit_usp_example.json` in ComfyUI for a complete working example.

## Support
- **Documentation**: See `XDIT_USP_README.md`
- **Testing**: Run `python3 test_xdit_usp.py`
- **Installation**: Run `./install_xdit_usp.sh` 