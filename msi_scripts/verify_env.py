#!/usr/bin/env python3
"""
Environment verification script for SDO Flare Prediction Model on MSI

This script checks for all required dependencies, tests GPU availability,
and validates the data directory structure before attempting to run the model.
"""

import os
import sys
import argparse
from importlib import import_module
import subprocess
import platform


def check_python_version():
    """Check if Python version is at least 3.8"""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"ERROR: Python 3.8+ required, found {major}.{minor}")
        return False
    print(f"✅ Python version {major}.{minor} is compatible")
    return True


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_packages = {
        "torch": "PyTorch is the main deep learning framework used",
        "torchvision": "Required for image transformations and models",
        "numpy": "Required for numerical operations",
        "pandas": "Required for data handling",
        "scikit-learn": "Required for metrics and data splitting",
        "matplotlib": "Required for visualization",
        "opencv-python": "Required for image processing (cv2)",
        "pytorch_lightning": "Required for training utilities",
        "captum": "Required for model interpretation",
        "yaml": "Required for configuration handling",
    }
    
    # Add additional common aliases
    package_aliases = {
        "opencv-python": ["cv2"],
        "scikit-learn": ["sklearn"],
        "pytorch_lightning": ["pytorch_lightning", "lightning"],
        "yaml": ["yaml", "pyyaml"],
    }
    
    missing_packages = []
    all_success = True
    
    for package, description in required_packages.items():
        aliases = [package]
        if package in package_aliases:
            aliases = package_aliases[package]
        
        found = False
        for alias in aliases:
            try:
                if alias == "cv2":
                    # Special case for OpenCV
                    import cv2
                    print(f"✅ {package} (version {cv2.__version__}) - {description}")
                    found = True
                    break
                else:
                    # Try to import and get version for other packages
                    module = import_module(alias)
                    version = getattr(module, "__version__", "unknown version")
                    print(f"✅ {package} (version {version}) - {description}")
                    found = True
                    break
            except ImportError:
                continue
        
        if not found:
            print(f"❌ {package} - MISSING - {description}")
            missing_packages.append(package)
            all_success = False
    
    if missing_packages:
        print("\nMissing dependencies. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return all_success


def check_gpu():
    """Check if CUDA is available and test PyTorch GPU detection"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                print(f"✅ CUDA available with {device_count} device(s)")
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    print(f"  - GPU {i}: {device_name}")
                
                # Run a small test tensor on the GPU
                try:
                    x = torch.rand(1000, 1000, device="cuda")
                    y = torch.matmul(x, x.t())
                    del x, y
                    print("✅ GPU test computation successful")
                except Exception as e:
                    print(f"❌ GPU test computation failed: {e}")
                    return False
                return True
            else:
                print("❌ CUDA is available but no devices found")
                return False
        else:
            print("❌ CUDA is not available")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_data_directory(data_path):
    """Check if the data directory has the expected structure"""
    print(f"\nChecking data directory: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data directory {data_path} does not exist")
        return False
    
    # Check for expected subdirectories (training, test)
    training_dir = os.path.join(data_path, "training")
    test_dir = os.path.join(data_path, "test")
    
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Check for sample files
    train_samples = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
    if not train_samples:
        print("❌ No training samples found")
        return False
    
    print(f"✅ Found {len(train_samples)} training samples")
    
    test_samples = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not test_samples:
        print("❌ No test samples found")
        return False
    
    print(f"✅ Found {len(test_samples)} test samples")
    
    # Check a random sample to verify the expected structure
    sample_dir = os.path.join(training_dir, train_samples[0])
    timesteps = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
    
    if not timesteps:
        print(f"❌ No timestep directories found in sample {train_samples[0]}")
        return False
    
    print(f"✅ Sample structure looks correct with {len(timesteps)} timesteps")
    
    # Check for magnetogram and EUV files in the first timestep
    timestep_dir = os.path.join(sample_dir, timesteps[0])
    files = os.listdir(timestep_dir)
    
    mag_files = [f for f in files if 'magnetogram' in f.lower()]
    if not mag_files:
        print("❌ No magnetogram files found")
        return False
    
    euv_files = [f for f in files if any(f.endswith(f"_{wl}.jpg") or f.endswith(f"_{wl}.png") 
                                         for wl in ['94', '131', '171', '193', '211', '304', '335', '1700'])]
    if not euv_files:
        print("❌ No EUV files found")
        return False
    
    print(f"✅ Found magnetogram and {len(euv_files)} EUV files in sample timestep")
    
    return True


def check_system_stats():
    """Check system statistics including CPU, memory, and disk space"""
    print("\nChecking system statistics...")

    # System info
    system = platform.system()
    print(f"OS: {platform.platform()}")

    # CPU info
    import psutil
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_count} physical cores, {cpu_logical} logical cores")

    # Memory info
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024 ** 3)
    print(f"RAM: {ram_gb:.1f} GB (total)")
    
    # Disk info
    disk = psutil.disk_usage(os.getcwd())
    disk_gb = disk.total / (1024 ** 3)
    free_gb = disk.free / (1024 ** 3)
    print(f"Disk: {disk_gb:.1f} GB (total), {free_gb:.1f} GB (free)")

    return True


def test_model_imports():
    """Test importing the model code"""
    print("\nTesting model imports...")
    
    try:
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == "msi_scripts":
            # We're in the scripts directory, go up
            os.chdir(os.path.dirname(current_dir))
        
        # Add current dir to path if not already there
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Try importing model components
        from models import SolarFlareModel
        print("✅ Successfully imported SolarFlareModel")
        
        from models import SolarFlareLoss
        print("✅ Successfully imported SolarFlareLoss")
        
        from data.preprocessing import get_data_loaders
        print("✅ Successfully imported data preprocessing utilities")
        
        return True
    except Exception as e:
        print(f"❌ Error importing model code: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify the environment for SDO Flare Prediction Model")
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDO Flare Prediction Model - Environment Verification")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Core checks
    all_checks_passed &= check_python_version()
    all_checks_passed &= check_dependencies()
    all_checks_passed &= check_gpu()
    all_checks_passed &= check_system_stats()
    all_checks_passed &= test_model_imports()
    
    # Data checks (if path provided)
    if args.data_path:
        all_checks_passed &= check_data_directory(args.data_path)
    else:
        print("\nData path not provided. Skipping data directory checks.")
        print("To check the data directory, run with:")
        print("python verify_env.py --data_path /path/to/your/data")
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ All checks passed! Environment is ready for training.")
    else:
        print("❌ Some checks failed. Please address the issues above before training.")
    print("=" * 60)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 