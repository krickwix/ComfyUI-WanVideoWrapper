#!/usr/bin/env python3
"""
Fix HuggingFace Hub Dependency Issue
This script fixes the 'cached_download' import error by downgrading huggingface_hub
"""

import subprocess
import sys
import os

def check_huggingface_version():
    """Check current huggingface_hub version"""
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"Current huggingface_hub version: {version}")
        return version
    except ImportError:
        print("huggingface_hub not installed")
        return None

def fix_huggingface_dependency():
    """Fix the huggingface_hub dependency issue"""
    print("🔧 Fixing HuggingFace Hub dependency issue...")
    
    current_version = check_huggingface_version()
    
    if current_version:
        # Check if version is too new (causing the cached_download issue)
        try:
            from packaging import version as pkg_version
            if pkg_version.parse(current_version) >= pkg_version.parse("0.20.0"):
                print(f"⚠️  Current version {current_version} is too new and causes import errors")
                print("   Downgrading to compatible version...")
                
                # Downgrade to a compatible version
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "huggingface_hub<0.20.0", "--force-reinstall"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Successfully downgraded huggingface_hub")
                    new_version = check_huggingface_version()
                    print(f"   New version: {new_version}")
                else:
                    print(f"❌ Failed to downgrade: {result.stderr}")
                    return False
            else:
                print(f"✅ Version {current_version} is compatible")
        except ImportError:
            print("⚠️  Could not parse version, attempting downgrade anyway...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "huggingface_hub<0.20.0", "--force-reinstall"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install: {result.stderr}")
                return False
    else:
        # Install if not present
        print("📦 Installing huggingface_hub...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "huggingface_hub<0.20.0"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install: {result.stderr}")
            return False
    
    # Test the fix
    print("\n🧪 Testing the fix...")
    try:
        from huggingface_hub import cached_download
        print("✅ cached_download import successful!")
        return True
    except ImportError as e:
        print(f"❌ Still cannot import cached_download: {e}")
        return False

def main():
    """Main function"""
    print("HuggingFace Hub Dependency Fix")
    print("=" * 40)
    
    success = fix_huggingface_dependency()
    
    if success:
        print("\n🎉 Fix applied successfully!")
        print("\nNext steps:")
        print("1. Restart ComfyUI completely")
        print("2. Check that WanVideoWrapper loads without errors")
        print("3. Look for DistriFusion nodes under 'WanVideoWrapper/DistriFusion'")
    else:
        print("\n❌ Fix failed. Please try manually:")
        print("pip install huggingface_hub<0.20.0 --force-reinstall")
        print("Then restart ComfyUI")

if __name__ == "__main__":
    main() 