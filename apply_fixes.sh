#!/bin/bash

# Script to apply WanVideo fixes to ComfyUI pod
# This script copies our fixed files to the running ComfyUI pod

echo "🔧 Applying WanVideo fixes to ComfyUI pod..."

# Get the current ComfyUI pod name
POD_NAME=$(kubectl get pods | grep comfy | grep Running | awk '{print $1}' | head -1)

if [ -z "$POD_NAME" ]; then
    echo "❌ No running ComfyUI pod found"
    exit 1
fi

echo "📦 Found pod: $POD_NAME"

# Copy the fixed files
echo "📋 Copying fixed model.py..."
kubectl cp wanvideo/modules/model.py $POD_NAME:/opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper/wanvideo/modules/model.py

echo "📋 Copying multi_gpu_utils.py..."
kubectl cp multi_gpu_utils.py $POD_NAME:/opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper/

echo "📋 Copying nodes_model_loading.py..."
kubectl cp nodes_model_loading.py $POD_NAME:/opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper/

echo "📋 Copying documentation files..."
kubectl cp MULTI_GPU_GUIDE.md $POD_NAME:/opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper/ 2>/dev/null || true

# Clear Python cache
echo "🧹 Clearing Python cache..."
kubectl exec $POD_NAME -- find /opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper -name "*.pyc" -delete 2>/dev/null || true
kubectl exec $POD_NAME -- find /opt/comfyui/custom_nodes/ComfyUI-WanVideoWrapper -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Fixes applied successfully!"
echo "🔄 You may need to restart ComfyUI in the pod to see the changes:"
echo "   kubectl exec $POD_NAME -- pkill -f 'python.*main.py'"
echo ""
echo "📊 To check the logs:"
echo "   kubectl logs $POD_NAME --tail=50" 