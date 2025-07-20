#!/bin/bash

# XDit+USP Installation Script for ComfyUI-WanVideoWrapper
# This script installs the required dependencies for distributed inference

set -e

echo "=========================================="
echo "XDit+USP Distributed Inference Installer"
echo "=========================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
    PIP_CMD="pip"
elif command -v pip3 &> /dev/null; then
    echo "✓ Using pip3"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "✓ Using pip"
    PIP_CMD="pip"
else
    echo "✗ No pip found. Please install pip first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python version: $PYTHON_VERSION"

# Check CUDA availability
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        echo "✓ CUDA is available"
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        echo "✓ CUDA version: $CUDA_VERSION"
    else
        echo "⚠️  CUDA is not available. Distributed inference may not work properly."
    fi
else
    echo "⚠️  PyTorch not found. Will install it as part of the dependencies."
fi

# Install base requirements first
echo ""
echo "Installing base requirements..."
$PIP_CMD install -r requirements.txt

# Install XDit+USP specific dependencies
echo ""
echo "Installing XDit+USP dependencies..."

# Try to install xdit
echo "Installing xdit..."
if $PIP_CMD install xdit>=0.1.0; then
    echo "✓ xdit installed successfully"
else
    echo "⚠️  Failed to install xdit. You may need to install it manually."
    echo "   Try: pip install xdit>=0.1.0"
fi

# Try to install usp
echo "Installing usp..."
if $PIP_CMD install usp>=0.1.0; then
    echo "✓ usp installed successfully"
else
    echo "⚠️  Failed to install usp. You may need to install it manually."
    echo "   Try: pip install usp>=0.1.0"
fi

# Verify installation
echo ""
echo "Verifying installation..."

# Test imports
if python3 -c "import xdit; print('✓ xdit imported successfully')" 2>/dev/null; then
    echo "✓ xdit is available"
else
    echo "✗ xdit import failed"
fi

if python3 -c "import usp; print('✓ usp imported successfully')" 2>/dev/null; then
    echo "✓ usp is available"
else
    echo "✗ usp import failed"
fi

# Test our implementation
if python3 -c "from nodes_xdit_usp_loader import XDitUSPConfig; print('✓ XDit+USP nodes imported successfully')" 2>/dev/null; then
    echo "✓ XDit+USP nodes are available"
else
    echo "✗ XDit+USP nodes import failed"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart ComfyUI"
echo "2. Look for 'XDit+USP Config' and 'XDit+USP WanVideo Model Loader' nodes"
echo "3. Check XDIT_USP_README.md for usage instructions"
echo "4. Try the example workflow: example_workflows/xdit_usp_example.json"
echo ""
echo "For troubleshooting, run: python3 test_xdit_usp.py" 