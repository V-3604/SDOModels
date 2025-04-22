#!/bin/bash

# Default values
JOB_SCRIPT="train_model.sh"
DATA_DIR="/path/to/default/data"
USE_TMPDIR="false"
MULTI_GPU="false"
SKIP_VERIFY="false"

# Help function
function show_help {
    echo "Usage: ./submit_job.sh [options]"
    echo ""
    echo "Options:"
    echo "  -s, --script SCRIPT    SLURM script to submit (default: train_model.sh)"
    echo "  -d, --data DIR         Path to data directory"
    echo "  -t, --tmp              Use TMPDIR for faster data access"
    echo "  -m, --multi-gpu        Use multi-GPU training script"
    echo "  --skip-verify          Skip environment verification check"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  ./submit_job.sh --data /home/user/sdo_data --tmp --multi-gpu"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--script)
            JOB_SCRIPT="$2"
            shift
            shift
            ;;
        -d|--data)
            DATA_DIR="$2"
            shift
            shift
            ;;
        -t|--tmp)
            USE_TMPDIR="true"
            shift
            ;;
        -m|--multi-gpu)
            MULTI_GPU="true"
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set job script based on multi-GPU flag
if [ "$MULTI_GPU" = "true" ] && [ "$JOB_SCRIPT" = "train_model.sh" ]; then
    JOB_SCRIPT="train_multi_gpu.sh"
    echo "Using multi-GPU training script: $JOB_SCRIPT"
fi

# Verify the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Run environment verification unless skipped
if [ "$SKIP_VERIFY" = "false" ]; then
    echo "Running environment verification check..."
    
    # Make sure we have psutil for system checks
    if ! python -c "import psutil" &> /dev/null; then
        echo "Installing psutil for system checks..."
        pip install psutil --user
    fi
    
    # Run the verification script
    python msi_scripts/verify_env.py --data_path "$DATA_DIR"
    VERIFY_STATUS=$?
    
    if [ $VERIFY_STATUS -ne 0 ]; then
        echo "Environment verification failed. Fix issues before submitting job."
        echo "To skip verification, use --skip-verify flag."
        exit 1
    fi
    
    echo "Environment verification passed. Proceeding with job submission."
else
    echo "Skipping environment verification check."
fi

# Export environment variables for the SLURM script
export DATA_DIR="$DATA_DIR"
export USE_TMPDIR="$USE_TMPDIR"

echo "Submitting job with the following parameters:"
echo "  Script: $JOB_SCRIPT"
echo "  Data directory: $DATA_DIR"
echo "  Use TMPDIR: $USE_TMPDIR"

# Submit the job
sbatch msi_scripts/$JOB_SCRIPT

echo "Job submitted! Check status with 'squeue -u $USER'" 