# MSI Scripts for SDO Flare Prediction Model

This directory contains scripts for running the SDO Flare Prediction Model on the Minnesota Supercomputing Institute (MSI) A100 GPUs.

## Available Scripts

1. `train_model.sh`: SLURM script for training on a single A100 GPU
2. `train_multi_gpu.sh`: SLURM script for distributed training on 4 A100 GPUs
3. `submit_job.sh`: Helper script for submitting jobs with the correct environment variables

## Usage

### Basic Job Submission

To submit a training job using a single GPU:

```bash
# Make scripts executable
chmod +x submit_job.sh

# Submit a job with default parameters
./submit_job.sh --data /path/to/sdo/data
```

### Using Multiple GPUs

To train using distributed training on 4 GPUs:

```bash
./submit_job.sh --data /path/to/sdo/data --multi-gpu
```

### Faster Data Access with TMPDIR

For better I/O performance, copy data to the node's local SSD:

```bash
./submit_job.sh --data /path/to/sdo/data --tmp
```

### Full Example

```bash
./submit_job.sh --data /home/user/projects/sdo_data --tmp --multi-gpu
```

## Customizing SLURM Scripts

You can modify the SLURM scripts directly to customize:

1. Resource requests (memory, CPUs, GPUs)
2. Time limits
3. Email notifications
4. Model hyperparameters

For example, to change the learning rate, edit the appropriate script and modify:

```bash
python main.py train \
  # other parameters...
  --learning_rate 5e-7 \  # Changed from 1e-6
  # other parameters...
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job logs:
```bash
cat jobid.out  # Replace jobid with your actual job ID
```

Monitor GPU usage:
```bash
srun --jobid=jobid --pty nvidia-smi
```

## Common Issues

1. **Out of Memory**: Reduce batch size or model complexity
2. **Job Time Limit**: Increase the `--time` parameter in the SLURM script
3. **CUDA Out of Memory**: Use mixed precision and/or gradient accumulation

## Additional Resources

- [MSI User Guide](https://www.msi.umn.edu/content/user-guide)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 