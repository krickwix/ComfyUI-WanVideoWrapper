#!/usr/bin/env python3
"""
DistriFusion Node Verification Script
This script verifies that DistriFusion nodes are properly registered in ComfyUI
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch not available")
    
    try:
        import torch.distributed
        print("✅ PyTorch distributed available")
    except ImportError:
        missing_deps.append("torch.distributed")
        print("❌ PyTorch distributed not available")
    
    try:
        import diffusers
        print("✅ Diffusers available")
    except ImportError:
        missing_deps.append("diffusers")
        print("❌ Diffusers not available")
    
    try:
        import comfy
        print("✅ ComfyUI available")
    except ImportError:
        missing_deps.append("comfy")
        print("❌ ComfyUI not available")
    
    return missing_deps

def verify_distrifusion_imports():
    """Verify DistriFusion module imports"""
    print("\n=== DistriFusion Import Verification ===")
    
    try:
        from distrifusion import __version__
        print(f"✅ DistriFusion package imported (v{__version__})")
    except ImportError as e:
        print(f"❌ DistriFusion package import failed: {e}")
        return False
    
    # Test individual components
    components = [
        ("DistriFusionModelLoader", "distrifusion.distributed_model_loader"),
        ("DistriFusionDistributionConfig", "distrifusion.distributed_model_loader"),
        ("DistriFusionWanModel", "distrifusion.distrifusion_wrapper"),
        ("PatchManager", "distrifusion.patch_manager"),
        ("AsyncPatchCommunicator", "distrifusion.communication"),
    ]
    
    available_components = []
    for component_name, module_path in components:
        try:
            module = __import__(module_path, fromlist=[component_name])
            component = getattr(module, component_name)
            available_components.append(component_name)
            print(f"✅ {component_name} available")
        except Exception as e:
            print(f"❌ {component_name} failed: {e}")
    
    return len(available_components) > 0

def verify_node_registration():
    """Verify that DistriFusion nodes are registered in ComfyUI"""
    print("\n=== Node Registration Verification ===")
    
    try:
        # Import the main nodes module
        import nodes
        
        if not hasattr(nodes, 'NODE_CLASS_MAPPINGS'):
            print("❌ NODE_CLASS_MAPPINGS not found in nodes module")
            return False
        
        # Look for DistriFusion nodes
        node_mappings = nodes.NODE_CLASS_MAPPINGS
        distrifusion_nodes = {k: v for k, v in node_mappings.items() if 'DistriFusion' in k}
        
        print(f"📋 Found {len(distrifusion_nodes)} DistriFusion nodes:")
        for node_name, node_class in distrifusion_nodes.items():
            print(f"   ✅ {node_name} -> {node_class.__name__}")
        
        # Check display names
        if hasattr(nodes, 'NODE_DISPLAY_NAME_MAPPINGS'):
            display_mappings = nodes.NODE_DISPLAY_NAME_MAPPINGS
            distrifusion_displays = {k: v for k, v in display_mappings.items() if 'DistriFusion' in k}
            
            print(f"🏷️  Found {len(distrifusion_displays)} DistriFusion display names:")
            for node_name, display_name in distrifusion_displays.items():
                print(f"   📝 {node_name} -> '{display_name}'")
        
        # Test node instantiation
        print("\n🔧 Testing node instantiation:")
        for node_name, node_class in distrifusion_nodes.items():
            try:
                # Test INPUT_TYPES method
                input_types = node_class.INPUT_TYPES()
                required_inputs = len(input_types.get('required', {}))
                optional_inputs = len(input_types.get('optional', {}))
                print(f"   ✅ {node_name}: {required_inputs} required, {optional_inputs} optional inputs")
            except Exception as e:
                print(f"   ❌ {node_name}: Failed to get INPUT_TYPES - {e}")
        
        return len(distrifusion_nodes) > 0
        
    except Exception as e:
        print(f"❌ Node registration check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("DistriFusion ComfyUI Node Verification")
    print("=" * 50)
    
    # Check dependencies
    print("\n🔍 Checking Dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Install these to enable DistriFusion nodes:")
        for dep in missing_deps:
            if dep == "torch":
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            elif dep == "diffusers":
                print("   pip install diffusers")
            elif dep == "comfy":
                print("   DistriFusion must be used within ComfyUI environment")
        
        print("\n   In a ComfyUI environment with dependencies, DistriFusion nodes should appear under:")
        print("   'WanVideoWrapper/DistriFusion' category")
        return
    
    # Check DistriFusion imports
    distrifusion_available = verify_distrifusion_imports()
    
    if not distrifusion_available:
        print("\n❌ DistriFusion components not available")
        print("   Check the import error messages above")
        return
    
    # Check node registration
    nodes_registered = verify_node_registration()
    
    if nodes_registered:
        print("\n🎉 SUCCESS: DistriFusion nodes should be available in ComfyUI!")
        print("\nLook for these nodes under 'WanVideoWrapper/DistriFusion' category:")
        print("   - DistriFusion Model Loader (recommended)")
        print("   - DistriFusion Distribution Config")
        print("   - DistriFusion WanVideo Model Loader (Legacy)")
        print("   - DistriFusion WanVideo Sampler")
        print("   - DistriFusion Setup")
        print("   - DistriFusion Status")
    else:
        print("\n❌ DistriFusion nodes not properly registered")
        print("   Check ComfyUI console for error messages during startup")

if __name__ == "__main__":
    main() 