"""
DistriFusion Launcher
Multi-GPU launcher for distributed WanVideo inference
"""

import os
import sys
import subprocess
import argparse
import torch
import signal
from typing import List, Optional
from utils import log


class DistriFusionLauncher:
    """
    Launcher for multi-GPU DistriFusion inference
    Handles process spawning and distributed environment setup
    """
    
    def __init__(self):
        self.processes = []
        self.master_port = None
        
    def launch_distributed(self,
                          script_path: str,
                          world_size: int,
                          master_addr: str = "localhost",
                          master_port: Optional[int] = None,
                          backend: str = "nccl",
                          additional_args: Optional[List[str]] = None) -> bool:
        """
        Launch distributed DistriFusion inference
        
        Args:
            script_path: Path to the ComfyUI script or workflow
            world_size: Number of GPUs/processes to use
            master_addr: Master node address
            master_port: Master node port (auto-assigned if None)
            backend: Communication backend (nccl/gloo)
            additional_args: Additional arguments to pass to each process
            
        Returns:
            True if launch successful, False otherwise
        """
        # Validate inputs
        if world_size > torch.cuda.device_count():
            log.error(f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
            return False
        
        if world_size < 2:
            log.error("DistriFusion requires at least 2 GPUs")
            return False
        
        # Auto-assign port if not provided
        if master_port is None:
            master_port = self._find_free_port()
        
        self.master_port = master_port
        
        # Set up signal handling for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        log.info(f"Launching DistriFusion with {world_size} GPUs")
        log.info(f"Master: {master_addr}:{master_port}")
        log.info(f"Backend: {backend}")
        
        # Launch processes
        try:
            for rank in range(world_size):
                process = self._launch_process(
                    script_path=script_path,
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    backend=backend,
                    additional_args=additional_args or []
                )
                
                if process is None:
                    log.error(f"Failed to launch process for rank {rank}")
                    self.cleanup()
                    return False
                
                self.processes.append(process)
                log.info(f"Launched process for rank {rank} (PID: {process.pid})")
            
            # Wait for all processes to complete
            return self._wait_for_completion()
            
        except Exception as e:
            log.error(f"Failed to launch distributed processes: {e}")
            self.cleanup()
            return False
    
    def _launch_process(self,
                       script_path: str,
                       rank: int,
                       world_size: int,
                       master_addr: str,
                       master_port: int,
                       backend: str,
                       additional_args: List[str]) -> Optional[subprocess.Popen]:
        """Launch a single process for given rank"""
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': str(rank),
            'MASTER_ADDR': master_addr,
            'MASTER_PORT': str(master_port),
            'WORLD_SIZE': str(world_size),
            'RANK': str(rank),
            'LOCAL_RANK': str(rank),
            'NCCL_DEBUG': 'INFO',  # Enable NCCL debugging
            'TORCH_DISTRIBUTED_DEBUG': 'INFO'
        })
        
        # Build command
        cmd = [
            sys.executable,
            script_path,
            f"--rank={rank}",
            f"--world-size={world_size}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
            f"--backend={backend}",
        ] + additional_args
        
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return process
            
        except Exception as e:
            log.error(f"Failed to start process for rank {rank}: {e}")
            return None
    
    def _wait_for_completion(self) -> bool:
        """Wait for all processes to complete"""
        try:
            success = True
            
            for i, process in enumerate(self.processes):
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    log.error(f"Process {i} failed with return code {process.returncode}")
                    log.error(f"STDERR: {stderr}")
                    success = False
                else:
                    log.info(f"Process {i} completed successfully")
                
                # Log output
                if stdout:
                    log.info(f"Process {i} STDOUT:\n{stdout}")
                if stderr and process.returncode == 0:
                    log.warning(f"Process {i} STDERR:\n{stderr}")
            
            return success
            
        except Exception as e:
            log.error(f"Error waiting for processes: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        log.info(f"Received signal {signum}, cleaning up processes...")
        self.cleanup()
        sys.exit(1)
    
    def cleanup(self):
        """Cleanup all spawned processes"""
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Process still running
                log.info(f"Terminating process {i} (PID: {process.pid})")
                process.terminate()
                
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log.warning(f"Force killing process {i}")
                    process.kill()
        
        self.processes.clear()
    
    def _find_free_port(self) -> int:
        """Find a free port for master communication"""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        
        return port


def create_distrifusion_script(workflow_path: str, 
                              output_script: str = "distrifusion_inference.py") -> str:
    """
    Create a DistriFusion inference script from a ComfyUI workflow
    
    Args:
        workflow_path: Path to ComfyUI workflow JSON
        output_script: Output script path
        
    Returns:
        Path to created script
    """
    script_template = '''#!/usr/bin/env python3
"""
Auto-generated DistriFusion inference script
"""

import argparse
import json
import torch
import torch.distributed as dist
import os
import sys

# Add ComfyUI path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ComfyUI_WanVideoWrapper.distrifusion import DistriFusionWanVideoModelLoader, DistriFusionWanVideoSampler
from ComfyUI_WanVideoWrapper.distrifusion.communication import DistributedManager


def main():
    parser = argparse.ArgumentParser(description="DistriFusion Distributed Inference")
    parser.add_argument("--rank", type=int, required=True, help="Process rank")
    parser.add_argument("--world-size", type=int, required=True, help="World size")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=int, required=True, help="Master port")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend")
    parser.add_argument("--workflow", type=str, default="{workflow_path}", help="Workflow path")
    
    args = parser.parse_args()
    
    # Initialize distributed environment
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.backend,
            rank=args.rank,
            world_size=args.world_size
        )
    
    print(f"Initialized rank {{args.rank}}/{{args.world_size}}")
    
    # Load workflow
    with open(args.workflow, 'r') as f:
        workflow = json.load(f)
    
    # Execute workflow with DistriFusion
    try:
        # This is a simplified example - actual implementation would parse
        # the workflow and execute nodes appropriately
        execute_workflow(workflow, args.rank, args.world_size)
        
    except Exception as e:
        print(f"Execution failed on rank {{args.rank}}: {{e}}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def execute_workflow(workflow, rank, world_size):
    """Execute workflow with DistriFusion"""
    # This is a placeholder - actual implementation would:
    # 1. Parse workflow nodes
    # 2. Find DistriFusion model loader and sampler nodes
    # 3. Execute them with proper distributed setup
    # 4. Handle data distribution and collection
    
    print(f"Executing workflow on rank {{rank}}/{{world_size}}")
    # Add actual execution logic here
    

if __name__ == "__main__":
    main()
'''
    
    # Write script
    script_content = script_template.format(workflow_path=workflow_path)
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_script, 0o755)
    
    log.info(f"Created DistriFusion script: {output_script}")
    return output_script


def main():
    """Command line interface for DistriFusion launcher"""
    parser = argparse.ArgumentParser(description="Launch DistriFusion distributed inference")
    parser.add_argument("--script", type=str, required=True, help="Script or workflow to run")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=int, help="Master port (auto-assigned if not specified)")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Communication backend")
    parser.add_argument("--create-script", action="store_true", help="Create script from workflow")
    parser.add_argument("--workflow", type=str, help="Workflow JSON file (for --create-script)")
    
    args = parser.parse_args()
    
    # Create script from workflow if requested
    if args.create_script:
        if not args.workflow:
            print("Error: --workflow required when using --create-script")
            return False
        
        script_path = create_distrifusion_script(args.workflow)
        print(f"Created script: {script_path}")
        return True
    
    # Launch distributed inference
    launcher = DistriFusionLauncher()
    
    try:
        success = launcher.launch_distributed(
            script_path=args.script,
            world_size=args.gpus,
            master_addr=args.master_addr,
            master_port=args.master_port,
            backend=args.backend
        )
        
        if success:
            print("DistriFusion inference completed successfully")
        else:
            print("DistriFusion inference failed")
            
        return success
        
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
        launcher.cleanup()
        return False
    except Exception as e:
        print(f"Error: {e}")
        launcher.cleanup()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 