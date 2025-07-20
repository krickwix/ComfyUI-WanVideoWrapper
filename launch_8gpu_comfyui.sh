#!/bin/bash

# Launch ComfyUI with 8-GPU distributed inference support
# This script sets up the proper environment variables for multi-GPU usage

echo "🚀 Launching ComfyUI with 8-GPU distributed inference support..."

# Set distributed environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8
export RANK=0

# Set PyTorch distributed variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "📋 Environment variables set:"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   RANK: $RANK"
echo "   NCCL_DEBUG: $NCCL_DEBUG"

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Launch ComfyUI
echo "🎯 Starting ComfyUI..."
echo "   - Load the workflow: example_workflows/8gpu_distributed_workflow.json"
echo "   - Make sure your model files are in the correct locations"
echo "   - The distributed config is set to use all 8 GPUs"
echo ""

# Change to ComfyUI directory if it exists
if [ -d "/opt/comfyui" ]; then
    cd /opt/comfyui
    echo "📁 Changed to ComfyUI directory: /opt/comfyui"
elif [ -d "../ComfyUI" ]; then
    cd ../ComfyUI
    echo "📁 Changed to ComfyUI directory: ../ComfyUI"
else
    echo "⚠️  ComfyUI directory not found. Please run this script from the ComfyUI directory."
    echo "   Or modify the script to point to your ComfyUI installation."
fi

# Launch ComfyUI with distributed support
python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header 