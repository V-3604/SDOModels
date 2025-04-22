# SDO Solar Flare Prediction Model

A state-of-the-art deep learning model for predicting solar flare peak flux using data from the Solar Dynamics Observatory (SDO).

## Overview

This repository contains a comprehensive implementation of a solar flare prediction model that leverages multi-modal, spatiotemporal deep learning to predict solar flare intensities within a 24-hour prediction window. The model uses magnetogram and multi-channel EUV images from SDO to achieve state-of-the-art performance in both regression (peak flux prediction) and classification tasks.

## Key Features

- **Advanced Architecture**: Combines DenseNet121 backbones, bidirectional LSTM/GRU, and Transformer components for optimal feature extraction and temporal modeling
- **Multi-Modal Fusion**: Implements sophisticated attention mechanisms to prioritize magnetogram and key EUV channels (94Å and 211Å)
- **Temporal Evolution**: Explicitly models the temporal evolution across 4 sequential timesteps (12h, 5h, 1.5h, and 10min before event)
- **Uncertainty Quantification**: Provides uncertainty estimates for predictions, crucial for operational usage
- **Multi-Task Learning**: Simultaneously optimizes for regression (peak flux) and classification (C vs. quiet, M vs. C, M vs. quiet) 
- **Physics-Informed**: Incorporates physics-based regularization to enhance model robustness
- **Comprehensive Evaluation**: Evaluated using Mean Absolute Error (MAE) for regression and True Skill Statistic (TSS) for classification
- **Explainable AI**: Integrates modern interpretation techniques for model explainability

## Model Architecture

The model architecture consists of several key components:

1. **Backbone Networks**: DenseNet121 for both magnetogram and EUV image processing
2. **Channel Attention**: Dynamic weighting of input channels with emphasis on magnetogram, 94Å, and 211Å channels
3. **Temporal Modeling**: Bidirectional LSTM/GRU with optional Transformer enhancement for capturing temporal dependencies
4. **Multi-Task Heads**: Specialized output heads for peak flux regression (with uncertainty) and flare classification tasks

## Performance Targets

The model aims to achieve the following performance metrics:

- **Classification Performance**:
  - M vs. 0 classification: TSS > 0.86
  - C vs. 0 classification: TSS > 0.58
  - M vs. C classification: TSS > 0.58

- **Regression Performance**:
  - Overall MAE < 3.6e-5

## Repository Structure

```
SDOModels/
├── data/                      # Data loading and preprocessing
│   ├── preprocessing.py       # Data preparation utilities
│   └── __init__.py
├── models/                    # Neural network architecture
│   ├── backbone.py            # CNN backbone models
│   ├── model.py               # Full solar flare prediction model
│   ├── temporal.py            # Temporal modeling components
│   └── __init__.py
├── training/                  # Training scripts
│   ├── train.py               # Main training loop
│   └── __init__.py
├── evaluation/                # Evaluation utilities
│   ├── evaluate.py            # Model evaluation
│   └── __init__.py
├── visualization/             # Visualization tools
│   ├── visualize.py           # Visualization utilities
│   └── __init__.py
├── notebooks/                 # Jupyter notebooks
│   └── sdo_model_demo.ipynb   # Demo notebook
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
├── LICENSE                    # License information
└── README.md                  # Project documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/SDOModels.git
   cd SDOModels
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

```bash
python main.py train --data_path /path/to/sdo/data --batch_size 32 --temporal_type lstm
```

For more options, run:
```bash
python main.py train --help
```

### Evaluating a Trained Model

```bash
python main.py evaluate --checkpoint_path /path/to/checkpoint.ckpt --data_path /path/to/test/data
```

For more options, run:
```bash
python main.py evaluate --help
```

### Using the Demo Notebook

The repository includes a Jupyter notebook (`notebooks/sdo_model_demo.ipynb`) that demonstrates the key functionality of the model:

```bash
jupyter notebook notebooks/sdo_model_demo.ipynb
```

## Data Format

The model expects SDO data in the following format:

- Magnetogram images: Line-of-sight magnetograms from HMI
- EUV images: Multi-channel observations from AIA (focusing on 94Å, 131Å, 171Å, 193Å, 211Å, 304Å, and 335Å channels)
- Each active region should have 4 sequential timesteps (12h, 5h, 1.5h, and 10min before the prediction window)

## GPU Resources

For optimal performance, the model should be trained on a GPU with at least 12GB of memory. The implementation includes mixed-precision training to optimize GPU memory usage.

## Running on MSI (Minnesota Supercomputing Institute)

This project supports running on MSI's high-performance computing resources, specifically optimized for their NVIDIA A100 GPUs.

### MSI Setup

1. **Accessing MSI**:
   ```bash
   ssh username@login.msi.umn.edu
   ```
   Note: You'll need to set up Duo 2FA for MSI access.

2. **Environment Setup**:
   ```bash
   module load python3/3.9.3
   module load cuda/11.3.0
   module load pytorch/1.10.0
   
   # Or create a conda environment
   module load conda
   conda create -n sdoflare python=3.9 pytorch=1.10.0 cudatoolkit=11.3 -c pytorch
   conda activate sdoflare
   ```

3. **Clone and Install**:
   ```bash
   cd /path/to/your/project/directory
   git clone https://github.com/username/SDOModels.git
   cd SDOModels
   pip install -r requirements.txt
   ```

### SLURM Job Submission

Create a SLURM job script (e.g., `train_model.sh`):

```bash
#!/bin/bash -l
#SBATCH --job-name=sdo_flare
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --tmp=10G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -e %j.err
#SBATCH -o %j.out

cd /path/to/your/SDOModels

module load python3/3.9.3
module load cuda/11.3.0
module load pytorch/1.10.0

# Alternatively, if using conda:
# module load conda
# conda activate sdoflare

# Run training
python main.py train \
  --data_path /path/to/sdo/data \
  --batch_size 64 \
  --mixed_precision \
  --temporal_type lstm \
  --experiment_name a100_training_run
```

Submit your job:
```bash
sbatch train_model.sh
```

### Multi-GPU Training

For distributed training across multiple A100 GPUs, adjust your SLURM script:

```bash
#SBATCH --gres=gpu:a100:4  # Request 4 GPUs

# Run with distributed training
python -m torch.distributed.launch --nproc_per_node=4 main.py train \
  --data_path /path/to/sdo/data \
  --batch_size 32 \
  --mixed_precision \
  --distributed
```

### Monitoring Jobs

Check job status:
```bash
squeue -u username
```

View job output in real-time:
```bash
tail -f jobid.out
```

Check GPU utilization:
```bash
srun --jobid=jobid --pty nvidia-smi
```

### Storage Considerations

MSI has various storage options with different quotas and performance characteristics:
- Home directory: Limited quota but backed up
- Project space: Larger quota for shared work
- Scratch space: Highest performance but no backup

For large datasets, use:
```bash
# Add to your SLURM script
cp -r /path/to/project/data $TMPDIR
python main.py train --data_path $TMPDIR/data
```

## Citation

If you use this code or model in your research, please cite:

```
@misc{sdo_flare_model,
  author = {Author},
  title = {SDO Solar Flare Prediction Model},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/username/SDOModels}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation draws inspiration from recent research in solar flare prediction and deep learning methodologies:

- [SDOBenchmark](https://github.com/SolarDynamicsObservatory/SDOBenchmark) for benchmark datasets and evaluation metrics
- Advanced deep learning architectures for multimodal fusion and spatiotemporal modeling
- Uncertainty quantification methodologies for reliable prediction
- Explainable AI techniques for model interpretation