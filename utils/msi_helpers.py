"""
MSI Supercomputing specific utilities for SDO model training
"""
import os
import platform
import socket
import torch
import yaml
import subprocess
from pathlib import Path

def get_msi_node_info():
    """Get information about the current MSI node"""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }
    return info

def print_system_info():
    """Print system information for debugging"""
    info = get_msi_node_info()
    print("\n" + "="*50)
    print("MSI SYSTEM INFORMATION")
    print("="*50)
    print(f"Hostname: {info['hostname']}")
    print(f"Platform: {info['platform']}")
    print(f"CPU Count: {info['cpu_count']}")
    print(f"GPU Count: {info['gpu_count']}")
    if info['gpu_count'] > 0:
        for i, name in enumerate(info['gpu_names']):
            print(f"GPU {i}: {name}")
    print("="*50 + "\n")

def load_msi_config(config_path="configs/msi.yaml"):
    """Load MSI configuration and format username"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace {username} with actual username in paths
    username = os.environ.get('USER')
    for key in ['path', 'train_metadata', 'test_metadata']:
        if key in config['data'] and '{username}' in config['data'][key]:
            config['data'][key] = config['data'][key].replace('{username}', username)
    
    return config

def check_dataset_availability(config):
    """Check if dataset is available at the configured path"""
    dataset_path = config['data']['path']
    train_meta = config['data']['train_metadata']
    test_meta = config['data']['test_metadata']
    
    dataset_path_obj = Path(dataset_path)
    train_meta_obj = Path(train_meta)
    test_meta_obj = Path(test_meta)
    
    if not dataset_path_obj.exists():
        print(f"WARNING: Dataset path {dataset_path} does not exist!")
        return False
    
    if not train_meta_obj.exists():
        print(f"WARNING: Training metadata {train_meta} does not exist!")
        return False
    
    if not test_meta_obj.exists():
        print(f"WARNING: Test metadata {test_meta} does not exist!")
        return False
    
    print(f"Dataset verified at {dataset_path}")
    sample_count = len([d for d in os.listdir(dataset_path_obj / "training") 
                        if (dataset_path_obj / "training" / d).is_dir()])
    print(f"Training samples found: {sample_count}")
    
    return True

def check_gpu_utilization():
    """Check current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True)
        print("\nCurrent GPU Utilization:")
        for i, line in enumerate(result.stdout.strip().split('\n')):
            print(f"GPU {i}: {line}")
    except Exception as e:
        print(f"Unable to query GPU utilization: {e}")

def setup_msi_environment():
    """Setup MSI-specific environment variables and paths"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    # Check if running in SLURM environment
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        print(f"Running as SLURM job: {job_id}")
        
        # Set recommended environment variables for PyTorch
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        
        # Get GPU device details if available
        if torch.cuda.is_available():
            print(f"CUDA Devices visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
            
    return job_id is not None 