#!/bin/bash
#SBATCH --job-name=sdo_train
#SBATCH --output=logs/sdo_train_%j.out
#SBATCH --error=logs/sdo_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -p a100-4  # Adjust based on available MSI partitions

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Directory: $(pwd)"

# Load necessary modules
module load python3/3.9.3
module load cuda/11.7.0
module load cudnn/8.4.1.50_cuda11.6

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TORCH_DISTRIBUTED_DEBUG=INFO

# Print GPU information
nvidia-smi

# Run the training script
echo "Starting training..."
srun python main.py --config configs/msi.yaml --command train --experiment_name "sdo_model_msi_$(date +%Y%m%d_%H%M%S)"

echo "Job completed: $(date)" 