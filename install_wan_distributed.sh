#!/bin/bash

# Wan2.1 Distributed Inference Installation Script
# This script installs dependencies for Wan2.1 distributed inference

set -e

echo "=========================================="
echo "Wan2.1 Distributed Inference Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "nodes_xdit_usp_loader.py" ]; then
    print_error "This script must be run from the ComfyUI-WanVideoWrapper directory"
    exit 1
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python $python_version is compatible (requires >= $required_version)"
else
    print_error "Python $python_version is not compatible (requires >= $required_version)"
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -n1)
    print_success "CUDA $cuda_version detected"
    
    # Check GPU count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_status "Found $gpu_count GPU(s):"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while IFS=, read -r name memory; do
        print_status "  $name ($memory)"
    done
else
    print_warning "nvidia-smi not found. CUDA may not be available."
fi

# Check PyTorch installation
print_status "Checking PyTorch installation..."
if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_success "PyTorch $pytorch_version is installed"
    
    # Check CUDA support in PyTorch
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "PyTorch CUDA support is available"
    else
        print_warning "PyTorch CUDA support is not available"
    fi
    
    # Check distributed support
    if python3 -c "import torch.distributed" 2>/dev/null; then
        print_success "PyTorch distributed support is available"
    else
        print_warning "PyTorch distributed support is not available"
    fi
else
    print_error "PyTorch is not installed"
    print_status "Installing PyTorch..."
    
    # Install PyTorch with CUDA support
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU version..."
        pip3 install torch torchvision torchaudio
    fi
fi

# Install xfuser (optional, for context parallel)
print_status "Checking xfuser installation..."
if python3 -c "import xfuser" 2>/dev/null; then
    xfuser_version=$(python3 -c "import xfuser; print(getattr(xfuser, '__version__', 'unknown'))" 2>/dev/null)
    print_success "xfuser $xfuser_version is installed"
else
    print_warning "xfuser is not installed (optional for context parallel support)"
    read -p "Do you want to install xfuser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing xfuser..."
        pip3 install xfuser>=0.1.0
        print_success "xfuser installed successfully"
    else
        print_warning "xfuser installation skipped. Context parallel features will not be available."
    fi
fi

# Install other dependencies
print_status "Installing additional dependencies..."
pip3 install -r requirements.txt

# Verify installation
print_status "Verifying installation..."

# Test imports
print_status "Testing imports..."
if python3 -c "
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
print('✓ PyTorch distributed components imported successfully')
" 2>/dev/null; then
    print_success "PyTorch distributed components are working"
else
    print_error "Failed to import PyTorch distributed components"
    exit 1
fi

# Test our implementation
print_status "Testing Wan2.1 distributed implementation..."
if python3 -c "
import sys
sys.path.append('.')
from nodes_xdit_usp_loader import WanDistributedConfig, WanDistributedModel
print('✓ Wan2.1 distributed components imported successfully')
" 2>/dev/null; then
    print_success "Wan2.1 distributed implementation is working"
else
    print_error "Failed to import Wan2.1 distributed components"
    exit 1
fi

# Test GPU availability
print_status "Testing GPU availability..."
if python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available with {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available')
" 2>/dev/null; then
    print_success "GPU test completed"
else
    print_warning "GPU test failed"
fi

# Create test script
print_status "Creating test script..."
cat > test_wan_distributed.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test for Wan2.1 distributed inference
"""

import torch
import sys

def quick_test():
    print("Wan2.1 Distributed Inference Quick Test")
    print("=" * 40)
    
    # Test basic imports
    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        print("✓ PyTorch distributed imports: OK")
    except ImportError as e:
        print(f"✗ PyTorch distributed imports: FAILED - {e}")
        return False
    
    # Test our implementation
    try:
        from nodes_xdit_usp_loader import WanDistributedConfig
        print("✓ Wan2.1 distributed imports: OK")
    except ImportError as e:
        print(f"✗ Wan2.1 distributed imports: FAILED - {e}")
        return False
    
    # Test GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU availability: {gpu_count} GPU(s) available")
    else:
        print("⚠️  GPU availability: CUDA not available")
    
    # Test configuration creation
    try:
        config = WanDistributedConfig(
            world_size=2,
            rank=0,
            use_fsdp=True,
            param_dtype=torch.bfloat16
        )
        print("✓ Configuration creation: OK")
    except Exception as e:
        print(f"✗ Configuration creation: FAILED - {e}")
        return False
    
    print("\n🎉 All tests passed! Wan2.1 distributed inference is ready.")
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
EOF

chmod +x test_wan_distributed.py
print_success "Test script created: test_wan_distributed.py"

# Final verification
print_status "Running final verification..."
if python3 test_wan_distributed.py; then
    print_success "Final verification passed"
else
    print_error "Final verification failed"
    exit 1
fi

echo
echo "=========================================="
print_success "Installation completed successfully!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Start ComfyUI"
echo "2. Look for 'Wan2.1 Distributed Config' and 'Wan2.1 Distributed Model Loader' nodes"
echo "3. Load the example workflow: example_workflows/wan_distributed_example.json"
echo "4. Configure your distributed settings and enjoy multi-GPU inference!"
echo
echo "Documentation:"
echo "- WAN_DISTRIBUTED_README.md - Comprehensive guide"
echo "- test_wan_distributed.py - Run tests"
echo
echo "For support, check the documentation or create an issue on GitHub." 