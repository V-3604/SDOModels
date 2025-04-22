import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, roc_auc_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import SolarFlareModel, SolarFlareLoss, PhysicsInformedRegularization
from data.preprocessing import get_data_loaders, SDODataAugmentation


class SolarFlareModule(pl.LightningModule):
    """
    PyTorch Lightning module for solar flare prediction
    """
    def __init__(self,
                 model_config: dict,
                 loss_config: dict,
                 optimizer_config: dict,
                 data_config: dict):
        """
        Initialize lightning module
        
        Args:
            model_config: Configuration for model architecture
            loss_config: Configuration for loss function
            optimizer_config: Configuration for optimizer
            data_config: Configuration for data loading
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = SolarFlareModel(**model_config)
        
        # Create loss function
        self.criterion = SolarFlareLoss(**loss_config)
        
        # Create physics-informed regularization
        self.physics_reg = PhysicsInformedRegularization(
            weight=loss_config.get('physics_reg_weight', 0.1),
            spatial_order=loss_config.get('spatial_order', 2)
        )
        
        # Store configurations
        self.optimizer_config = optimizer_config
        self.data_config = data_config
        
        # Initialize metrics
        self.best_val_mae = float('inf')
        self.best_val_tss = -float('inf')
        
        # Add gradient norm tracking
        self.grad_norm = 0.0
        
        # Initialize learning rate history
        self.lr_history = []
        
        # Initialize validation metric history for stability check
        self.val_loss_history = []
        self.val_stable_iters = 0
        
    def forward(self, magnetogram, euv):
        """
        Forward pass
        
        Args:
            magnetogram: Magnetogram tensor
            euv: EUV tensor
            
        Returns:
            Model predictions
        """
        return self.model(magnetogram, euv)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            Optimizer and scheduler configuration
        """
        # Extract optimizer parameters
        lr = self.optimizer_config.get('lr', 5e-5)  # Increased from 1e-6
        weight_decay = self.optimizer_config.get('weight_decay', 0.001)  # Reduced from 0.01
        scheduler_type = self.optimizer_config.get('scheduler', 'cosine')
        use_warmup = self.optimizer_config.get('use_warmup', True)
        warmup_epochs = self.optimizer_config.get('warmup_epochs', 5)
        
        # Create parameter groups with different weight decay
        # Apply less weight decay to bias and layer norm parameters
        no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Create optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=1e-8  # Increased epsilon for stability
        )
        
        # Create scheduler
        if scheduler_type == 'cosine':
            t_0 = self.optimizer_config.get('t_0', 20)  # Increased from 10
            t_mult = self.optimizer_config.get('t_mult', 2)
            eta_min = self.optimizer_config.get('eta_min', 1e-7)
            
            if use_warmup:
                # Create warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda epoch: min(1.0, epoch / warmup_epochs)
                )
                
                # Main cosine scheduler after warmup
                main_scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=t_0,
                    T_mult=t_mult,
                    eta_min=eta_min
                )
                
                # Combined scheduler
                scheduler = {
                    'scheduler': pl.utilities.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers=[warmup_scheduler, main_scheduler],
                        milestones=[warmup_epochs]
                    ),
                    'interval': 'epoch',
                    'name': 'sequential_lr'
                }
            else:
                # Just use cosine annealing
                scheduler = {
                    'scheduler': CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=t_0,
                        T_mult=t_mult,
                        eta_min=eta_min
                    ),
                    'interval': 'epoch',
                    'name': 'cosine_lr'
                }
                
        elif scheduler_type == 'plateau':
            factor = self.optimizer_config.get('factor', 0.5)
            patience = self.optimizer_config.get('patience', 10)  # Increased from 5
            min_lr = self.optimizer_config.get('min_lr', 1e-7)
            
            if use_warmup:
                # Create warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda epoch: min(1.0, epoch / warmup_epochs)
                )
                
                # Main plateau scheduler after warmup
                main_scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                    verbose=True
                )
                
                # Combined scheduler
                scheduler = {
                    'scheduler': pl.utilities.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers=[warmup_scheduler, main_scheduler],
                        milestones=[warmup_epochs]
                    ),
                    'interval': 'epoch',
                    'monitor': 'val_loss',
                    'name': 'sequential_lr'
                }
            else:
                # Just use plateau scheduler
                scheduler = {
                    'scheduler': ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=factor,
                        patience=patience,
                        min_lr=min_lr,
                        verbose=True
                    ),
                    'interval': 'epoch',
                    'monitor': 'val/mae',
                    'name': 'plateau_lr'
                }
        else:
            scheduler = None
            
        return [optimizer] if scheduler is None else [optimizer], [scheduler]
    
    def on_train_epoch_start(self):
        """Callback at the start of each training epoch"""
        # Log current learning rate
        for param_group in self.trainer.optimizers[0].param_groups:
            current_lr = param_group['lr']
            self.log('learning_rate', current_lr)
            self.lr_history.append(current_lr)
    
    def on_after_backward(self):
        """Callback after backward pass to compute gradient norm"""
        # Calculate gradient norm for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 
                                                  max_norm=float('inf'), 
                                                  norm_type=2)
        self.grad_norm = grad_norm.item()
        self.log('grad_norm', self.grad_norm, prog_bar=True)
        
        # Check for exploding gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            # Log warning
            print(f"Warning: Gradient norm is {grad_norm}. Skipping update.")
            # Zero out gradients to prevent parameter updates
            for param in self.parameters():
                param.grad = None
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Handle NaN/Inf values in inputs
        batch = self._handle_invalid_values(batch)
        
        # Get inputs and targets
        magnetogram = batch['magnetogram']
        euv = batch['euv']
        targets = batch['target']
        
        # Forward pass
        predictions = self(magnetogram, euv)
        
        # Compute main loss
        losses = self.criterion(predictions, targets)
        
        # Add physics-informed regularization if enabled
        if self.hparams.loss_config.get('use_physics_reg', True):
            physics_loss = self.physics_reg(magnetogram, predictions)
            losses['physics_reg'] = physics_loss
            losses['total'] += physics_loss
        
        # Adaptive gradient clipping based on recent gradient history
        clip_factor = 5.0
        if self.grad_norm > 0 and self.grad_norm > clip_factor * self.trainer.accumulate_grad_batches:
            # Use a more aggressive clip if gradients are large
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_factor)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            # Check for NaN/Inf values in losses
            if torch.isnan(loss_value) or torch.isinf(loss_value):
                print(f"Warning: {loss_name} is {loss_value}. Using a default value.")
                loss_value = torch.tensor(1.0, device=self.device)
                losses[loss_name] = loss_value
            
            self.log(f'train/{loss_name}', loss_value, prog_bar=loss_name=='total')
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary of validation metrics
        """
        # Handle NaN/Inf values in inputs
        batch = self._handle_invalid_values(batch)
        
        # Get inputs and targets
        magnetogram = batch['magnetogram']
        euv = batch['euv']
        targets = batch['target']
        metadata = batch['metadata']
        
        # Forward pass
        predictions = self(magnetogram, euv)
        
        # Compute loss
        losses = self.criterion(predictions, targets)
        
        # Add physics-informed regularization if enabled
        if self.hparams.loss_config.get('use_physics_reg', True):
            physics_loss = self.physics_reg(magnetogram, predictions)
            losses['physics_reg'] = physics_loss
            losses['total'] += physics_loss
        
        # Log losses
        for loss_name, loss_value in losses.items():
            # Check for NaN/Inf values in losses
            if torch.isnan(loss_value) or torch.isinf(loss_value):
                print(f"Warning: {loss_name} is {loss_value}. Using a default value.")
                loss_value = torch.tensor(1.0, device=self.device)
                losses[loss_name] = loss_value
                
            self.log(f'val/{loss_name}', loss_value, prog_bar=loss_name=='total', sync_dist=True)
        
        # Store predictions and targets for computing metrics at epoch end
        peak_flux_pred = predictions['peak_flux_mean'] if 'peak_flux_mean' in predictions else predictions['peak_flux']
        
        # Return predictions and targets for epoch end validation metrics
        return {
            'peak_flux_pred': peak_flux_pred.detach().cpu(),
            'peak_flux_true': targets['peak_flux'].detach().cpu(),
            'c_vs_0_pred': predictions.get('c_vs_0', None),
            'c_vs_0_true': targets['is_c_flare'],
            'm_vs_c_pred': predictions.get('m_vs_c', None),
            'm_vs_c_true': targets['is_m_flare'],
            'm_vs_0_pred': predictions.get('m_vs_0', None),
            'm_vs_0_true': targets['is_m_flare']
        }
    
    def on_validation_epoch_end(self):
        """Process validation outputs at the end of an epoch"""
        # Check if validation metric is stable
        if hasattr(self.trainer, 'callback_metrics') and 'val/mae' in self.trainer.callback_metrics:
            current_mae = self.trainer.callback_metrics['val/mae'].item()
            self.val_loss_history.append(current_mae)
            
            # Consider validation stable if MAE doesn't change much over 3 epochs
            if len(self.val_loss_history) >= 3:
                recent_vals = self.val_loss_history[-3:]
                # Calculate maximum relative change
                max_change = max([abs(recent_vals[i] - recent_vals[i-1]) / (recent_vals[i-1] + 1e-8) 
                                  for i in range(1, len(recent_vals))])
                
                if max_change < 0.01:  # Less than 1% change
                    self.val_stable_iters += 1
                else:
                    self.val_stable_iters = 0
                
                self.log('val_stability', self.val_stable_iters, prog_bar=True)
    
    def _handle_invalid_values(self, batch):
        """
        Handle NaN and Inf values in batch data
        
        Args:
            batch: Input batch
            
        Returns:
            Cleaned batch
        """
        for key in ['magnetogram', 'euv']:
            if key in batch:
                # Replace NaN and Inf with zeros or reasonable values
                batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=1.0, neginf=-1.0)
        
        if 'target' in batch:
            for key in batch['target']:
                batch['target'][key] = torch.nan_to_num(batch['target'][key], nan=0.0, posinf=1.0, neginf=0.0)
                
        return batch
    
    def validation_epoch_end(self, outputs):
        """
        Compute validation metrics at the end of epoch
        
        Args:
            outputs: List of outputs from validation_step
        """
        # Concatenate outputs from all validation batches
        peak_flux_pred = torch.cat([x['peak_flux_pred'] for x in outputs])
        peak_flux_true = torch.cat([x['peak_flux_true'] for x in outputs])
        
        # Compute MAE for regression
        mae = mean_absolute_error(
            peak_flux_true.numpy(),
            peak_flux_pred.numpy()
        )
        
        self.log('val/mae', mae, prog_bar=True, sync_dist=True)
        
        # Track best MAE
        if mae < self.best_val_mae:
            self.best_val_mae = mae
            self.log('val/best_mae', self.best_val_mae, prog_bar=True, sync_dist=True)
        
        # Compute classification metrics if multi-task is enabled
        if self.hparams.model_config.get('use_multi_task', True):
            # C vs 0 classification
            c_vs_0_pred = torch.cat([x['c_vs_0_pred'].detach().cpu() for x in outputs])
            c_vs_0_true = torch.cat([x['c_vs_0_true'].detach().cpu() for x in outputs])
            
            # M vs 0 classification
            m_vs_0_pred = torch.cat([x['m_vs_0_pred'].detach().cpu() for x in outputs])
            m_vs_0_true = torch.cat([x['m_vs_0_true'].detach().cpu() for x in outputs])
            
            # Compute True Skill Statistic (TSS) for M vs 0
            m_vs_0_pred_binary = (m_vs_0_pred > 0.5).float().numpy()
            m_vs_0_true_binary = m_vs_0_true.numpy()
            
            tn, fp, fn, tp = confusion_matrix(
                m_vs_0_true_binary, 
                m_vs_0_pred_binary,
                labels=[0, 1]
            ).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            
            # Log metrics
            self.log('val/m_vs_0_tss', tss, prog_bar=True, sync_dist=True)
            
            # Track best TSS
            if tss > self.best_val_tss:
                self.best_val_tss = tss
                self.log('val/best_tss', self.best_val_tss, prog_bar=True, sync_dist=True)
            
            # Compute AUC-ROC scores
            try:
                c_vs_0_auc = roc_auc_score(c_vs_0_true.numpy(), c_vs_0_pred.numpy())
                m_vs_0_auc = roc_auc_score(m_vs_0_true.numpy(), m_vs_0_pred.numpy())
                
                self.log('val/c_vs_0_auc', c_vs_0_auc, sync_dist=True)
                self.log('val/m_vs_0_auc', m_vs_0_auc, sync_dist=True)
            except Exception as e:
                self.print(f"Error computing AUC: {e}")
    
    def test_step(self, batch, batch_idx):
        """
        Test step (identical to validation step)
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary of test metrics
        """
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        """
        Compute test metrics at the end of epoch
        
        Args:
            outputs: List of outputs from test_step
        """
        # Similar to validation_epoch_end but for test set
        self.validation_epoch_end(outputs)
        
        # Additional test metrics could be implemented here
        
    def predict_step(self, batch, batch_idx):
        """
        Prediction step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Model predictions
        """
        # Get inputs
        magnetogram = batch['magnetogram']
        euv = batch['euv']
        
        # Forward pass
        return self(magnetogram, euv)


def get_model_config(args):
    """
    Get model configuration from arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Model configuration dictionary
    """
    return {
        'magnetogram_channels': args.magnetogram_channels,
        'euv_channels': args.euv_channels,
        'pretrained': args.pretrained,
        'freeze_backbones': args.freeze_backbones,
        'use_attention': args.use_attention,
        'fusion_method': args.fusion_method,
        'temporal_type': args.temporal_type,
        'temporal_hidden_size': args.temporal_hidden_size,
        'temporal_num_layers': args.temporal_num_layers,
        'dropout': args.dropout,
        'final_hidden_size': args.final_hidden_size,
        'use_uncertainty': args.use_uncertainty,
        'use_multi_task': args.use_multi_task
    }


def get_loss_config(args):
    """
    Get loss configuration from arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Loss configuration dictionary
    """
    return {
        'regression_weight': args.regression_weight,
        'c_vs_0_weight': args.c_vs_0_weight,
        'm_vs_c_weight': args.m_vs_c_weight,
        'm_vs_0_weight': args.m_vs_0_weight,
        'use_uncertainty': args.use_uncertainty,
        'uncertainty_weight': args.uncertainty_weight,
        'use_multi_task': args.use_multi_task,
        'physics_reg_weight': args.physics_reg_weight,
        'use_physics_reg': args.use_physics_reg,
        'spatial_order': args.spatial_order
    }


def get_optimizer_config(args):
    """
    Get optimizer configuration from arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Optimizer configuration dictionary
    """
    return {
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        't_0': args.scheduler_t0,
        't_mult': args.scheduler_t_mult,
        'eta_min': args.scheduler_eta_min,
        'factor': args.scheduler_factor,
        'patience': args.scheduler_patience,
        'min_lr': args.scheduler_min_lr,
        'use_warmup': args.use_warmup,
        'warmup_epochs': args.warmup_epochs
    }


def get_data_config(args):
    """
    Get data configuration from arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        Data configuration dictionary
    """
    return {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'sample_type': args.sample_type
    }


def train(args):
    """
    Main training function
    
    Args:
        args: Command-line arguments
    """
    # Set seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Configure mixed precision
    if args.mixed_precision:
        precision = 16
    else:
        precision = 32
    
    # Create data loaders
    data_loaders = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_type=args.sample_type
    )
    
    # Get configurations
    model_config = get_model_config(args)
    loss_config = get_loss_config(args)
    optimizer_config = get_optimizer_config(args)
    data_config = get_data_config(args)
    
    # Create Lightning module
    model = SolarFlareModule(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        data_config=data_config
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/mae',
            mode='min',
            save_top_k=3,
            filename='sdo-mae-{epoch:02d}-{val/mae:.4f}',
            save_last=True
        ),
        ModelCheckpoint(
            monitor='val/m_vs_0_tss',
            mode='max',
            save_top_k=3,
            filename='sdo-tss-{epoch:02d}-{val/m_vs_0_tss:.4f}'
        ),
        EarlyStopping(
            monitor='val/mae',
            mode='min',
            patience=args.early_stopping_patience,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=args.gradient_clip,
        deterministic=args.deterministic
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders['val']
    )
    
    # Test model
    trainer.test(
        model,
        dataloaders=data_loaders['test']
    )
    
    return model


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Train solar flare prediction model')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, default='data/sdo',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sample_type', type=str, default='oversampled',
                        choices=['all', 'balanced', 'oversampled'],
                        help='Sampling strategy for class imbalance')
    
    # Model configuration
    parser.add_argument('--magnetogram_channels', type=int, default=1,
                        help='Number of magnetogram channels')
    parser.add_argument('--euv_channels', type=int, default=7,
                        help='Number of EUV channels')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Whether to use pretrained backbones')
    parser.add_argument('--freeze_backbones', action='store_true', default=False,
                        help='Whether to freeze backbone layers')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Whether to use attention mechanism')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'sum', 'weighted_sum'],
                        help='Method to fuse features')
    parser.add_argument('--temporal_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer'],
                        help='Type of temporal modeling')
    parser.add_argument('--temporal_hidden_size', type=int, default=512,
                        help='Size of temporal model hidden state')
    parser.add_argument('--temporal_num_layers', type=int, default=2,
                        help='Number of temporal model layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--final_hidden_size', type=int, default=512,
                        help='Size of final hidden layer')
    parser.add_argument('--use_uncertainty', action='store_true', default=True,
                        help='Whether to predict uncertainty')
    parser.add_argument('--use_multi_task', action='store_true', default=True,
                        help='Whether to perform multi-task learning')
    
    # Loss configuration
    parser.add_argument('--regression_weight', type=float, default=1.0,
                        help='Weight for regression loss')
    parser.add_argument('--c_vs_0_weight', type=float, default=0.5,
                        help='Weight for C vs 0 classification loss')
    parser.add_argument('--m_vs_c_weight', type=float, default=0.5,
                        help='Weight for M vs C classification loss')
    parser.add_argument('--m_vs_0_weight', type=float, default=0.5,
                        help='Weight for M vs 0 classification loss')
    parser.add_argument('--uncertainty_weight', type=float, default=0.1,
                        help='Weight for uncertainty regularization')
    parser.add_argument('--physics_reg_weight', type=float, default=0.1,
                        help='Weight for physics-informed regularization')
    parser.add_argument('--use_physics_reg', action='store_true', default=True,
                        help='Whether to use physics-informed regularization')
    parser.add_argument('--spatial_order', type=int, default=2,
                        help='Spatial order for physics-informed regularization')
    
    # Optimizer configuration
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--scheduler_t0', type=int, default=10,
                        help='T_0 parameter for cosine annealing')
    parser.add_argument('--scheduler_t_mult', type=int, default=2,
                        help='T_mult parameter for cosine annealing')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-7,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                        help='Minimum learning rate for ReduceLROnPlateau')
    parser.add_argument('--use_warmup', action='store_true', default=True,
                        help='Whether to use learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # Training configuration
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Whether to use mixed precision training')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Whether to use deterministic algorithms')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging configuration
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--experiment_name', type=str, default='sdo_flare_model',
                        help='Name of experiment')
    
    args = parser.parse_args()
    
    # Train model
    model = train(args)
    
    return model


if __name__ == '__main__':
    main() 