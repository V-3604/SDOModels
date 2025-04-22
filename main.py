#!/usr/bin/env python3
"""
Main script for SDO Solar Flare Prediction Model

This script provides a convenient entry point for training and evaluating
the SDO solar flare prediction model.
"""

import os
import argparse
import subprocess
import sys
import yaml
from datetime import datetime

# Try to import MSI helpers
try:
    from utils.msi_helpers import (
        print_system_info, 
        load_msi_config, 
        check_dataset_availability, 
        check_gpu_utilization,
        setup_msi_environment
    )
    HAS_MSI_HELPERS = True
except ImportError:
    HAS_MSI_HELPERS = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SDO Solar Flare Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add configuration file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    
    # Data configuration
    train_parser.add_argument('--data_path', type=str, default='data/sdo',
                             help='Path to dataset')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Number of data loading workers')
    train_parser.add_argument('--sample_type', type=str, default='oversampled',
                             choices=['all', 'balanced', 'oversampled'],
                             help='Sampling strategy for class imbalance')
    
    # Model configuration
    train_parser.add_argument('--magnetogram_channels', type=int, default=1,
                             help='Number of magnetogram channels')
    train_parser.add_argument('--euv_channels', type=int, default=7,
                             help='Number of EUV channels')
    train_parser.add_argument('--pretrained', action='store_true', default=True,
                             help='Whether to use pretrained backbones')
    train_parser.add_argument('--freeze_backbones', action='store_true', default=False,
                             help='Whether to freeze backbone layers')
    train_parser.add_argument('--use_attention', action='store_true', default=True,
                             help='Whether to use attention mechanism')
    train_parser.add_argument('--fusion_method', type=str, default='concat',
                             choices=['concat', 'sum', 'weighted_sum'],
                             help='Method to fuse features')
    train_parser.add_argument('--temporal_type', type=str, default='lstm',
                             choices=['lstm', 'gru', 'transformer'],
                             help='Type of temporal modeling')
    train_parser.add_argument('--temporal_hidden_size', type=int, default=512,
                             help='Size of temporal model hidden state')
    train_parser.add_argument('--temporal_num_layers', type=int, default=2,
                             help='Number of temporal model layers')
    train_parser.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout probability')
    train_parser.add_argument('--final_hidden_size', type=int, default=512,
                             help='Size of final hidden layer')
    train_parser.add_argument('--use_uncertainty', action='store_true', default=True,
                             help='Whether to predict uncertainty')
    train_parser.add_argument('--use_multi_task', action='store_true', default=True,
                             help='Whether to perform multi-task learning')
    
    # Loss configuration
    train_parser.add_argument('--regression_weight', type=float, default=1.0,
                             help='Weight for regression loss')
    train_parser.add_argument('--c_vs_0_weight', type=float, default=0.5,
                             help='Weight for C vs 0 classification loss')
    train_parser.add_argument('--m_vs_c_weight', type=float, default=0.5,
                             help='Weight for M vs C classification loss')
    train_parser.add_argument('--m_vs_0_weight', type=float, default=0.5,
                             help='Weight for M vs 0 classification loss')
    train_parser.add_argument('--uncertainty_weight', type=float, default=0.1,
                             help='Weight for uncertainty regularization')
    train_parser.add_argument('--physics_reg_weight', type=float, default=0.1,
                             help='Weight for physics-informed regularization')
    train_parser.add_argument('--use_physics_reg', action='store_true', default=True,
                             help='Whether to use physics-informed regularization')
    
    # Optimizer configuration
    train_parser.add_argument('--learning_rate', type=float, default=1e-6,
                             help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=0.01,
                             help='Weight decay')
    train_parser.add_argument('--scheduler', type=str, default='cosine',
                             choices=['cosine', 'plateau', 'none'],
                             help='Learning rate scheduler')
    train_parser.add_argument('--scheduler_t0', type=int, default=10,
                             help='T_0 parameter for cosine annealing')
    train_parser.add_argument('--scheduler_t_mult', type=int, default=2,
                             help='T_mult parameter for cosine annealing')
    train_parser.add_argument('--scheduler_eta_min', type=float, default=1e-7,
                             help='Minimum learning rate for cosine annealing')
    train_parser.add_argument('--scheduler_factor', type=float, default=0.5,
                             help='Factor for ReduceLROnPlateau')
    train_parser.add_argument('--scheduler_patience', type=int, default=5,
                             help='Patience for ReduceLROnPlateau')
    train_parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                             help='Minimum learning rate for ReduceLROnPlateau')
    
    # Training configuration
    train_parser.add_argument('--max_epochs', type=int, default=100,
                             help='Maximum number of epochs')
    train_parser.add_argument('--early_stopping_patience', type=int, default=10,
                             help='Patience for early stopping')
    train_parser.add_argument('--gradient_clip', type=float, default=1.0,
                             help='Gradient clipping value')
    train_parser.add_argument('--mixed_precision', action='store_true', default=True,
                             help='Whether to use mixed precision training')
    train_parser.add_argument('--deterministic', action='store_true', default=False,
                             help='Whether to use deterministic algorithms')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Logging configuration
    train_parser.add_argument('--log_dir', type=str, default='logs',
                             help='Directory for logs')
    train_parser.add_argument('--experiment_name', type=str, default=None,
                             help='Name of experiment (default: auto-generated)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    
    # Model configuration
    eval_parser.add_argument('--checkpoint_path', type=str, default=None,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--data_path', type=str, default='data/sdo',
                            help='Path to dataset')
    eval_parser.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (default: auto-generated)')
    eval_parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size')
    eval_parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of data loading workers')
    eval_parser.add_argument('--sample_type', type=str, default='all',
                            choices=['all', 'balanced', 'oversampled'],
                            help='Sampling strategy for test data')
    eval_parser.add_argument('--do_interpretation', action='store_true',
                            help='Whether to perform model interpretation')
    eval_parser.add_argument('--num_interpretation_samples', type=int, default=10,
                            help='Number of samples to use for interpretation')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def convert_config_to_args(config, command='train'):
    """Convert config dictionary to command-line arguments"""
    # Create dummy namespace
    args = argparse.Namespace()
    args.command = command
    
    # Set default values
    if command == 'train':
        args.experiment_name = f"sdo_flare_{config['model']['temporal_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Data configuration
    data_config = config.get('data', {})
    args.data_path = data_config.get('path', 'data/sdo')
    args.batch_size = data_config.get('batch_size', 32)
    args.num_workers = data_config.get('num_workers', 4)
    args.sample_type = data_config.get('sample_type', 'oversampled')
    
    # Model configuration
    model_config = config.get('model', {})
    args.magnetogram_channels = model_config.get('magnetogram_channels', 1)
    args.euv_channels = model_config.get('euv_channels', 7)
    args.pretrained = model_config.get('pretrained', True)
    args.freeze_backbones = model_config.get('freeze_backbones', False)
    args.use_attention = model_config.get('use_attention', True)
    args.fusion_method = model_config.get('fusion_method', 'concat')
    args.temporal_type = model_config.get('temporal_type', 'lstm')
    args.temporal_hidden_size = model_config.get('temporal_hidden_size', 512)
    args.temporal_num_layers = model_config.get('temporal_num_layers', 2)
    args.dropout = model_config.get('dropout', 0.1)
    args.final_hidden_size = model_config.get('final_hidden_size', 512)
    args.use_uncertainty = model_config.get('use_uncertainty', True)
    args.use_multi_task = model_config.get('use_multi_task', True)
    
    # Loss configuration
    loss_config = config.get('loss', {})
    args.regression_weight = loss_config.get('regression_weight', 1.0)
    args.c_vs_0_weight = loss_config.get('c_vs_0_weight', 0.5)
    args.m_vs_c_weight = loss_config.get('m_vs_c_weight', 0.5)
    args.m_vs_0_weight = loss_config.get('m_vs_0_weight', 0.5)
    args.uncertainty_weight = loss_config.get('uncertainty_weight', 0.1)
    args.physics_reg_weight = loss_config.get('physics_reg_weight', 0.1)
    args.use_physics_reg = loss_config.get('use_physics_reg', True)
    
    # Optimizer configuration
    optimizer_config = config.get('optimizer', {})
    args.learning_rate = optimizer_config.get('lr', 1e-6)
    args.weight_decay = optimizer_config.get('weight_decay', 0.01)
    args.scheduler = optimizer_config.get('scheduler', 'cosine')
    args.scheduler_t0 = optimizer_config.get('t_0', 10)
    args.scheduler_t_mult = optimizer_config.get('t_mult', 2)
    args.scheduler_eta_min = optimizer_config.get('eta_min', 1e-7)
    args.scheduler_factor = optimizer_config.get('factor', 0.5)
    args.scheduler_patience = optimizer_config.get('patience', 5)
    args.scheduler_min_lr = optimizer_config.get('min_lr', 1e-7)
    args.gradient_clip = optimizer_config.get('gradient_clip_val', 1.0)
    
    # Training configuration
    training_config = config.get('training', {})
    args.max_epochs = training_config.get('max_epochs', 100)
    args.early_stopping_patience = training_config.get('early_stopping_patience', 10)
    args.mixed_precision = training_config.get('precision', 32) == 16
    args.deterministic = training_config.get('deterministic', False)
    args.seed = training_config.get('seed', 42)
    
    # Logging configuration
    logging_config = config.get('logging', {})
    args.log_dir = logging_config.get('log_dir', 'logs')
    
    return args


def train_model(args):
    """Run model training with the given arguments"""
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"sdo_flare_{args.temporal_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Print experiment information
    print(f"\nStarting experiment: {args.experiment_name}")
    print(f"Using data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    
    # Setup MSI environment if available
    if HAS_MSI_HELPERS:
        is_msi = setup_msi_environment()
        if is_msi:
            print("Running in MSI environment")
            check_gpu_utilization()
    
    # Convert args to command line arguments for the training script
    cmd_args = []
    for key, value in vars(args).items():
        if key not in ['command', 'config']:  # Skip some args
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f'--{key}')
            else:
                cmd_args.append(f'--{key}')
                cmd_args.append(str(value))
    
    # Run training script
    cmd = [sys.executable, 'training/train.py'] + cmd_args
    print(f"Running command: {' '.join(cmd)}")
    
    subprocess.run(cmd)


def evaluate_model(args):
    """Run model evaluation with the given arguments"""
    # Generate output directory if not provided
    if args.output_dir is None:
        checkpoint_name = os.path.basename(args.checkpoint_path).split('.')[0]
        args.output_dir = f"evaluation/results/{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MSI environment if available
    if HAS_MSI_HELPERS:
        is_msi = setup_msi_environment()
        if is_msi:
            print("Running in MSI environment")
            check_gpu_utilization()
    
    # Convert args to command line arguments for the evaluation script
    cmd_args = []
    for key, value in vars(args).items():
        if key not in ['command', 'config']:  # Skip some args
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f'--{key}')
            else:
                cmd_args.append(f'--{key}')
                cmd_args.append(str(value))
    
    # Run evaluation script
    cmd = [sys.executable, 'evaluation/evaluate.py'] + cmd_args
    print(f"Running command: {' '.join(cmd)}")
    
    subprocess.run(cmd)


def main():
    """Main function"""
    args = parse_args()
    
    # If config file is provided, load it and convert to args
    if args.config:
        print(f"Loading configuration from {args.config}")
        
        # Check if using MSI config
        if HAS_MSI_HELPERS and 'msi' in args.config:
            print_system_info()
            config = load_msi_config(args.config)
            check_dataset_availability(config)
        else:
            config = load_config(args.config)
            
        # Convert config to args
        config_args = convert_config_to_args(config, args.command)
        
        # Update args with values from config, but keep command-line overrides
        for key, value in vars(config_args).items():
            if key not in ['command']:
                # Only set the value if it wasn't explicitly provided in the command line
                if key not in sys.argv:
                    setattr(args, key, value)
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        print("Please specify a command. Run with --help for options.")
        sys.exit(1)


if __name__ == '__main__':
    main() 