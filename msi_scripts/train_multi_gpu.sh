#!/bin/bash -l
#SBATCH --job-name=sdo_multi_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --tmp=40G
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH -e %j.err
#SBATCH -o %j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@umn.edu

# Change to project directory
cd ${SLURM_SUBMIT_DIR}

# Load necessary modules
module load python3/3.9.3
module load cuda/11.3.0
module load pytorch/1.10.0

# Alternatively, if using conda:
# module load conda
# conda activate sdoflare

# Print system and GPU information for logging
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
nvidia-smi

# Optional: Copy data to temp directory for faster access
if [ "$USE_TMPDIR" = "true" ]; then
    echo "Copying data to TMPDIR for faster access..."
    cp -r $DATA_DIR $TMPDIR/data
    DATA_PATH=$TMPDIR/data
else
    DATA_PATH=$DATA_DIR
fi

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# Run distributed training with PyTorch distributed launch
python -m torch.distributed.launch --nproc_per_node=4 \
  main.py train \
  --data_path $DATA_PATH \
  --batch_size 32 \
  --mixed_precision \
  --temporal_type lstm \
  --use_attention \
  --use_multi_task \
  --use_uncertainty \
  --experiment_name a100_distributed_$(date +%Y%m%d_%H%M%S) \
  --max_epochs 100 \
  --learning_rate 1e-6 \
  --early_stopping_patience 10 \
  --scheduler cosine \
  --distributed

# Print job end information
echo "Job finished at $(date)" 