# Quick Start Guide

This guide will help you quickly get started with the SDO Solar Flare Prediction Model.

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU with at least 12GB memory (recommended)
- Basic knowledge of PyTorch and deep learning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/SDOModels.git
   cd SDOModels
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The model expects data in a specific format. You have two options:

### Option 1: Use the SDOBenchmark Dataset

The model is designed to work with the [SDOBenchmark dataset](https://github.com/SolarDynamicsObservatory/SDOBenchmark). Follow their instructions to download and prepare the data.

### Option 2: Prepare Your Own Data

If you have your own SDO data, you need to organize it in the following structure:

```
data/
├── train/
│   ├── magnetograms/
│   │   ├── ar_12345_20150215_120000.fits
│   │   └── ...
│   ├── euv/
│   │   ├── ar_12345_20150215_120000/
│   │   │   ├── 94.fits
│   │   │   ├── 131.fits
│   │   │   └── ...
│   │   └── ...
│   └── metadata.csv
├── val/
│   └── ...
└── test/
    └── ...
```

The `metadata.csv` file should contain the following columns:
- `ar_id`: Active region identifier
- `timestamp`: Observation timestamp
- `peak_flux`: Target peak flux value
- `flare_class`: Flare class (e.g., 'X', 'M', 'C', 'B', 'A', 'QUIET')

## Training a Model

### Basic Training

To train a model with default parameters:

```bash
python main.py train --data_path /path/to/data --experiment_name my_first_model
```

### Advanced Training Options

For more control over the training process:

```bash
python main.py train \
  --data_path /path/to/data \
  --experiment_name custom_model \
  --batch_size 64 \
  --temporal_type lstm \
  --learning_rate 1e-6 \
  --max_epochs 200 \
  --use_uncertainty \
  --use_multi_task \
  --mixed_precision
```

See all available options:

```bash
python main.py train --help
```

## Evaluating a Model

Once you have a trained model, you can evaluate it on your test set:

```bash
python main.py evaluate \
  --checkpoint_path logs/my_first_model/version_0/checkpoints/last.ckpt \
  --data_path /path/to/test/data \
  --do_interpretation
```

## Running the Demo Notebook

We provide a demo notebook that showcases the model's functionality:

```bash
jupyter notebook notebooks/sdo_model_demo.ipynb
```

## Example Workflows

### Training a Production Model

For the best performance, we recommend:

```bash
python main.py train \
  --data_path /path/to/data \
  --batch_size 32 \
  --temporal_type transformer \
  --temporal_hidden_size 768 \
  --temporal_num_layers 4 \
  --fusion_method weighted_sum \
  --learning_rate 5e-7 \
  --weight_decay 0.01 \
  --max_epochs 100 \
  --early_stopping_patience 15 \
  --mixed_precision \
  --use_uncertainty \
  --use_multi_task \
  --use_physics_reg \
  --c_vs_0_weight 0.7 \
  --m_vs_0_weight 0.7 \
  --physics_reg_weight 0.2
```

### Experimentation

For faster experimentation and iteration:

```bash
python main.py train \
  --data_path /path/to/data \
  --batch_size 64 \
  --temporal_type lstm \
  --temporal_hidden_size 256 \
  --temporal_num_layers 2 \
  --max_epochs 30 \
  --mixed_precision \
  --learning_rate 1e-6
```

## Troubleshooting

### Out of Memory Errors

If you encounter GPU memory issues:
- Reduce batch size
- Use mixed precision training
- Reduce model size (temporal_hidden_size, final_hidden_size)

### Training Instability

If training is unstable:
- Reduce learning rate
- Increase gradient clipping value
- Adjust loss weights

### Poor Performance

If model performance is poor:
- Ensure data is preprocessed correctly
- Try different temporal models (lstm, gru, transformer)
- Adjust class weights for imbalanced data

## Next Steps

After getting familiar with the basic usage, you can:

1. Experiment with different model architectures
2. Fine-tune hyperparameters for your specific dataset
3. Implement custom preprocessing for your data
4. Explore model interpretability with the visualization tools

For more detailed information, refer to the full [README.md](README.md). 