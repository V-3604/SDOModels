#!/bin/bash
# scripts/setup_msi.sh - Setup environment on MSI

# Print current node information
echo "Setting up environment on: $(hostname)"
echo "Current directory: $(pwd)"

# Load MSI modules
module load python3/3.9.3
module load cuda/11.7.0
module load cudnn/8.4.1.50_cuda11.6

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p logs
mkdir -p data/external
mkdir -p ~/scratch/SDOModels/data/

# Setup complete
echo "Setup complete. Environment is ready for training."
echo "Next steps:"
echo "1. Make sure your dataset is in ~/scratch/SDOModels/data/"
echo "2. Submit your training job with: sbatch scripts/submit_msi.sh" 