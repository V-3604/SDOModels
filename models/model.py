import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from models.backbone import DenseNetBackbone, MultiInputDenseNet, MultiModalFusion
from models.temporal import TemporalLSTM, TemporalGRU, TemporalTransformer, SpatioTemporalFusion


class UncertaintyHead(nn.Module):
    """
    Output head that predicts both mean and variance for uncertainty estimation
    """
    def __init__(self, 
                 in_features: int,
                 hidden_size: int = 256):
        """
        Initialize uncertainty head
        
        Args:
            in_features: Number of input features
            hidden_size: Size of hidden layer
        """
        super().__init__()
        
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Mean prediction
        self.mean_head = nn.Linear(hidden_size, 1)
        
        # Variance prediction (log variance for numerical stability)
        self.logvar_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of predicted mean and log variance
        """
        # Extract features
        features = self.features(x)
        
        # Predict mean
        mean = self.mean_head(features)
        
        # Predict log variance
        logvar = self.logvar_head(features)
        
        return mean, logvar


class ClassificationHead(nn.Module):
    """
    Output head for binary classification tasks
    """
    def __init__(self, 
                 in_features: int,
                 hidden_size: int = 256):
        """
        Initialize classification head
        
        Args:
            in_features: Number of input features
            hidden_size: Size of hidden layer
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Classification probability
        """
        return self.classifier(x)


class SolarFlareModel(nn.Module):
    """
    Full solar flare prediction model
    """
    def __init__(self,
                 magnetogram_channels: int = 1,
                 euv_channels: int = 7,
                 pretrained: bool = True,
                 freeze_backbones: bool = False,
                 use_attention: bool = True,
                 fusion_method: str = 'concat',
                 temporal_type: str = 'lstm',
                 temporal_hidden_size: int = 512,
                 temporal_num_layers: int = 2,
                 dropout: float = 0.1,
                 final_hidden_size: int = 512,
                 use_uncertainty: bool = True,
                 use_multi_task: bool = True):
        """
        Initialize solar flare prediction model
        
        Args:
            magnetogram_channels: Number of magnetogram channels
            euv_channels: Number of EUV channels
            pretrained: Whether to use pretrained backbone
            freeze_backbones: Whether to freeze backbone layers
            use_attention: Whether to use attention mechanism
            fusion_method: Method to fuse modalities ('concat', 'sum', 'weighted_sum')
            temporal_type: Type of temporal model ('lstm', 'gru', 'transformer')
            temporal_hidden_size: Size of temporal model hidden state
            temporal_num_layers: Number of temporal model layers
            dropout: Dropout probability
            final_hidden_size: Size of final hidden layer
            use_uncertainty: Whether to predict uncertainty
            use_multi_task: Whether to perform multi-task learning
        """
        super().__init__()
        
        # Feature extraction backbones
        self.cnn_backbone = MultiInputDenseNet(
            magnetogram_channels=magnetogram_channels,
            euv_channels=euv_channels,
            pretrained=pretrained,
            freeze_magnetogram=freeze_backbones,
            freeze_euv=freeze_backbones
        )
        
        # Average pooling for spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Multimodal fusion
        self.fusion = MultiModalFusion(
            magnetogram_channels=self.cnn_backbone.magnetogram_backbone.out_channels,
            euv_channels=self.cnn_backbone.euv_backbone.out_channels,
            use_attention=use_attention,
            fusion_method=fusion_method
        )
        
        # Spatio-temporal fusion
        self.temporal_fusion = SpatioTemporalFusion(
            spatial_channels=self.fusion.out_channels,
            temporal_type=temporal_type,
            hidden_size=temporal_hidden_size,
            num_layers=temporal_num_layers,
            dropout=dropout,
            bidirectional=True,
            transformer_nhead=8,
            transformer_dim_feedforward=temporal_hidden_size * 4
        )
        
        # Final feature dimension
        self.feature_dim = self.temporal_fusion.out_channels
        
        # Final feature extraction
        self.features = nn.Sequential(
            nn.Linear(self.feature_dim, final_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Determine output heads based on flags
        if use_uncertainty:
            self.regression_head = UncertaintyHead(final_hidden_size)
        else:
            self.regression_head = nn.Linear(final_hidden_size, 1)
        
        # Classification heads for multi-task learning
        if use_multi_task:
            # C vs 0 (C-class or above vs. quiet)
            self.c_vs_0_head = ClassificationHead(final_hidden_size)
            
            # M vs C (M-class or above vs. C-class)
            self.m_vs_c_head = ClassificationHead(final_hidden_size)
            
            # M vs 0 (M-class or above vs. quiet)
            self.m_vs_0_head = ClassificationHead(final_hidden_size)
        
        # Store configuration
        self.use_uncertainty = use_uncertainty
        self.use_multi_task = use_multi_task
        
    def forward(self, 
                magnetogram: torch.Tensor,
                euv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            magnetogram: Magnetogram tensor of shape (batch_size, time_steps, channels, height, width)
            euv: EUV tensor of shape (batch_size, time_steps, channels, height, width)
            
        Returns:
            Dictionary of predictions
        """
        batch_size, time_steps, mag_channels, height, width = magnetogram.shape
        _, _, euv_channels, _, _ = euv.shape
        
        # Process each timestep with CNN backbone
        mag_features_seq = []
        euv_features_seq = []
        
        for t in range(time_steps):
            # Extract features for current timestep
            mag_features, euv_features = self.cnn_backbone(
                magnetogram[:, t], euv[:, t]
            )
            
            # Apply global pooling
            mag_features = self.pool(mag_features)
            euv_features = self.pool(euv_features)
            
            # Remove spatial dimensions
            mag_features = mag_features.view(batch_size, -1)
            euv_features = euv_features.view(batch_size, -1)
            
            # Multimodal fusion
            fused_features = self.fusion(
                mag_features.view(batch_size, -1, 1, 1),
                euv_features.view(batch_size, -1, 1, 1)
            )
            
            # Remove spatial dimensions again
            fused_features = fused_features.view(batch_size, -1)
            
            # Collect features for sequence
            mag_features_seq.append(mag_features.unsqueeze(1))
            euv_features_seq.append(euv_features.unsqueeze(1))
        
        # Concatenate features across time dimension
        mag_features_seq = torch.cat(mag_features_seq, dim=1)
        euv_features_seq = torch.cat(euv_features_seq, dim=1)
        
        # Temporal modeling
        temporal_features = self.temporal_fusion(
            torch.cat([mag_features_seq, euv_features_seq], dim=2).unsqueeze(-1).unsqueeze(-1)
        )
        
        # Final feature extraction
        features = self.features(temporal_features)
        
        # Prepare output dictionary
        output = {}
        
        # Regression prediction
        if self.use_uncertainty:
            mean, logvar = self.regression_head(features)
            output['peak_flux_mean'] = mean
            output['peak_flux_logvar'] = logvar
        else:
            output['peak_flux'] = self.regression_head(features)
        
        # Classification predictions for multi-task learning
        if self.use_multi_task:
            output['c_vs_0'] = self.c_vs_0_head(features)
            output['m_vs_c'] = self.m_vs_c_head(features)
            output['m_vs_0'] = self.m_vs_0_head(features)
        
        return output


class SolarFlareLoss(nn.Module):
    """
    Multi-task loss function for solar flare prediction
    """
    def __init__(self,
                 regression_weight: float = 1.0,
                 c_vs_0_weight: float = 0.5,
                 m_vs_c_weight: float = 0.5,
                 m_vs_0_weight: float = 0.5,
                 use_uncertainty: bool = True,
                 uncertainty_weight: float = 0.1,
                 use_multi_task: bool = True,
                 dynamic_weighting: bool = True,
                 class_weights: Optional[Dict[str, torch.Tensor]] = None):
        """
        Initialize loss function
        
        Args:
            regression_weight: Weight for regression loss
            c_vs_0_weight: Weight for C vs 0 classification loss
            m_vs_c_weight: Weight for M vs C classification loss
            m_vs_0_weight: Weight for M vs 0 classification loss
            use_uncertainty: Whether to use uncertainty loss
            uncertainty_weight: Weight for uncertainty regularization
            use_multi_task: Whether to use multi-task learning
            dynamic_weighting: Whether to use dynamic task weighting
            class_weights: Optional weights for different classes in classification tasks
        """
        super().__init__()
        
        self.regression_weight = regression_weight
        self.c_vs_0_weight = c_vs_0_weight
        self.m_vs_c_weight = m_vs_c_weight
        self.m_vs_0_weight = m_vs_0_weight
        self.use_uncertainty = use_uncertainty
        self.uncertainty_weight = uncertainty_weight
        self.use_multi_task = use_multi_task
        self.dynamic_weighting = dynamic_weighting
        self.class_weights = class_weights or {}
        
        # Task weights will be dynamically adjusted during training if dynamic_weighting is True
        # Initialize learnable task weights
        if dynamic_weighting:
            self.log_task_weights = nn.Parameter(torch.zeros(4))  # [reg, c_vs_0, m_vs_c, m_vs_0]
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
    def focal_loss(self, pred, target, weight=None):
        """
        Focal loss for imbalanced classification
        
        Args:
            pred: Predicted probabilities
            target: Target labels
            weight: Optional class weights
            
        Returns:
            Focal loss
        """
        # Binary focal loss implementation
        pred_prob = pred.clamp(min=1e-7, max=1-1e-7)
        binary_target = target
        
        # Focal scaling factor
        if binary_target.bool().any():
            pt = torch.where(binary_target.bool(), pred_prob, 1-pred_prob)
            focal_weight = (1 - pt) ** self.focal_gamma
            
            # Class balance weight - can be further adjusted per batch
            alpha_weight = torch.ones_like(binary_target)
            alpha_t = self.focal_alpha
            alpha_weight[binary_target.bool()] = alpha_t
            alpha_weight[~binary_target.bool()] = 1 - alpha_t
            
            # Calculate loss with focal and class weights
            loss = -alpha_weight * focal_weight * (
                binary_target * torch.log(pred_prob) + 
                (1 - binary_target) * torch.log(1 - pred_prob)
            )
            
            # Apply additional weight if provided
            if weight is not None:
                loss = loss * weight
                
            return loss.mean()
        else:
            # Handle edge case with no positive samples
            return F.binary_cross_entropy(pred_prob, binary_target, weight=weight)
    
    def get_class_weights(self, task, targets):
        """
        Compute class weights dynamically based on batch statistics
        
        Args:
            task: Task name
            targets: Target labels
            
        Returns:
            Class weights tensor
        """
        if self.dynamic_weighting and targets.numel() > 0:
            pos_ratio = targets.float().mean()
            # Avoid division by zero or extreme weights
            pos_ratio = torch.clamp(pos_ratio, min=0.05, max=0.95)
            neg_ratio = 1.0 - pos_ratio
            
            # Inversely proportional to class frequency
            pos_weight = neg_ratio / pos_ratio
            
            # Create a weight tensor for the batch
            weights = torch.ones_like(targets)
            weights[targets.bool()] = pos_weight
            
            return weights
        elif task in self.class_weights:
            return self.class_weights[task]
        else:
            return None
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        
        # Get task weights if using dynamic weighting
        if self.dynamic_weighting:
            # Softmax the log weights to get normalized task weights
            task_weights = F.softmax(self.log_task_weights, dim=0)
            reg_weight = task_weights[0] * self.regression_weight
            c_vs_0_weight = task_weights[1] * self.c_vs_0_weight
            m_vs_c_weight = task_weights[2] * self.m_vs_c_weight
            m_vs_0_weight = task_weights[3] * self.m_vs_0_weight
            
            # Log the dynamic weights for monitoring
            losses['reg_weight'] = reg_weight.detach()
            losses['c_vs_0_weight'] = c_vs_0_weight.detach()
            losses['m_vs_c_weight'] = m_vs_c_weight.detach()
            losses['m_vs_0_weight'] = m_vs_0_weight.detach()
        else:
            reg_weight = self.regression_weight
            c_vs_0_weight = self.c_vs_0_weight
            m_vs_c_weight = self.m_vs_c_weight
            m_vs_0_weight = self.m_vs_0_weight
        
        # Regression loss with optional uncertainty
        if self.use_uncertainty:
            # Get mean and log variance predictions
            mean = predictions['peak_flux_mean']
            logvar = predictions['peak_flux_logvar']
            
            # Get target values
            target = targets['peak_flux']
            
            # Improved uncertainty loss - use beta distribution for flux values
            # Since our values are now log-transformed to be positive
            # Compute negative log-likelihood with precision
            precision = torch.exp(-logvar)
            mse = self.mse_loss(mean, target)
            reg_loss = torch.mean(mse * precision + logvar)
            
            # Uncertainty regularization using KL divergence
            # Regularize to avoid trivial solutions (very high variance)
            # This essentially penalizes very large variances
            kl_term = 0.5 * torch.mean(
                -1.0 - logvar + mean.pow(2) + logvar.exp()
            )
            uncertainty_reg = self.uncertainty_weight * kl_term
            
            # Store individual losses
            losses['regression'] = reg_loss
            losses['uncertainty_reg'] = uncertainty_reg
            
            # Add to total loss
            total_loss = reg_weight * reg_loss + uncertainty_reg
        else:
            # Simple MSE loss
            reg_loss = F.mse_loss(predictions['peak_flux'], targets['peak_flux'])
            losses['regression'] = reg_loss
            total_loss = reg_weight * reg_loss
        
        # Classification losses for multi-task learning
        if self.use_multi_task:
            # Get dynamic class weights for each task
            c_vs_0_class_weight = self.get_class_weights('c_vs_0', targets['is_c_flare'])
            m_vs_0_class_weight = self.get_class_weights('m_vs_0', targets['is_m_flare'])
            
            # C vs 0 classification using focal loss for class imbalance
            c_vs_0_loss = self.focal_loss(
                predictions['c_vs_0'], 
                targets['is_c_flare'],
                weight=c_vs_0_class_weight
            )
            losses['c_vs_0'] = c_vs_0_loss
            total_loss += c_vs_0_weight * c_vs_0_loss
            
            # M vs 0 classification
            m_vs_0_loss = self.focal_loss(
                predictions['m_vs_0'], 
                targets['is_m_flare'],
                weight=m_vs_0_class_weight
            )
            losses['m_vs_0'] = m_vs_0_loss
            total_loss += m_vs_0_weight * m_vs_0_loss
            
            # Filter samples for M vs C classification (only C-class flares and above)
            m_vs_c_mask = targets['is_c_flare'].bool()
            if torch.sum(m_vs_c_mask) > 0:
                # Get filtered predictions and targets
                m_vs_c_pred = predictions['m_vs_c'][m_vs_c_mask]
                m_vs_c_targ = targets['is_m_flare'][m_vs_c_mask]
                
                # Get class weights for the filtered subset
                m_vs_c_class_weight = self.get_class_weights('m_vs_c', m_vs_c_targ)
                
                # Compute focal loss
                m_vs_c_loss = self.focal_loss(
                    m_vs_c_pred, m_vs_c_targ, weight=m_vs_c_class_weight
                )
                losses['m_vs_c'] = m_vs_c_loss
                total_loss += m_vs_c_weight * m_vs_c_loss
        
        # Store total loss
        losses['total'] = total_loss
        
        return losses


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-informed regularization for solar flare prediction
    """
    def __init__(self, weight: float = 0.1, spatial_order: int = 2):
        """
        Initialize physics-informed regularization
        
        Args:
            weight: Weight for regularization term
            spatial_order: Order of spatial derivatives (1 or 2)
        """
        super().__init__()
        
        self.weight = weight
        self.spatial_order = spatial_order
        
        # Define Sobel and Laplacian kernels for higher-order finite difference
        # Sobel kernels for first derivatives
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # Laplacian kernel for second derivatives
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # Higher-order Sobel kernels (5x5) for more accurate gradients
        self.sobel_x_5x5 = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 5, 5) / 60.0
        
        self.sobel_y_5x5 = torch.tensor([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32).reshape(1, 1, 5, 5) / 60.0
        
    def compute_free_energy_proxy(self, mag_field):
        """
        Compute a proxy for magnetic free energy
        
        Free energy is the difference between the non-potential (observed) field energy
        and the potential field energy. We approximate this using the current density.
        
        Args:
            mag_field: Magnetogram tensor (B, C, H, W)
            
        Returns:
            Energy proxy tensor
        """
        device = mag_field.device
        
        # Create kernels on the correct device
        if self.spatial_order == 2:
            # Higher-order gradient calculation (5x5 kernel)
            sobel_x = self.sobel_x_5x5.to(device)
            sobel_y = self.sobel_y_5x5.to(device)
            padding = 2
        else:
            # Standard gradient calculation (3x3 kernel)
            sobel_x = self.sobel_x.to(device)
            sobel_y = self.sobel_y.to(device)
            padding = 1
        
        # Calculate spatial gradients (∂B/∂x, ∂B/∂y)
        grad_x = F.conv2d(mag_field, sobel_x, padding=padding)
        grad_y = F.conv2d(mag_field, sobel_y, padding=padding)
        
        # Compute current density approximation J ~ curl(B)
        # In 2D, this is |∂B/∂x| + |∂B/∂y|
        current_density = torch.abs(grad_x) + torch.abs(grad_y)
        
        # Calculate the square of gradient magnitude (proxy for energy density)
        grad_magnitude_sq = grad_x**2 + grad_y**2
        
        # High current regions typically correlate with free energy
        # Combine magnitude and non-potentiality measure
        energy_proxy = current_density * torch.sqrt(grad_magnitude_sq)
        
        # Compute a scalar measure by taking spatial mean
        return torch.mean(energy_proxy)
        
    def compute_magnetic_complexity(self, mag_field):
        """
        Compute a measure of magnetic field complexity
        
        This captures the structural complexity of the field using
        the Laplacian (second derivative) which highlights areas of
        rapid field changes.
        
        Args:
            mag_field: Magnetogram tensor (B, C, H, W)
            
        Returns:
            Complexity measure
        """
        device = mag_field.device
        laplacian = self.laplacian.to(device)
        
        # Apply Laplacian to measure field curvature
        field_curvature = F.conv2d(mag_field, laplacian, padding=1)
        
        # Areas with high curvature correlate with complex topology
        complexity_measure = torch.mean(torch.abs(field_curvature))
        
        return complexity_measure
        
    def compute_energy_growth_rate(self, mag_sequence):
        """
        Compute the temporal growth rate of magnetic energy
        
        Rapid energy buildup is associated with flare likelihood
        
        Args:
            mag_sequence: Sequence of magnetogram tensors (B, T, C, H, W)
            
        Returns:
            Energy growth rate measure
        """
        # Extract magnetograms from consecutive timesteps
        batch_size, time_steps, channels, height, width = mag_sequence.shape
        
        # Need at least 2 timesteps
        if time_steps < 2:
            return torch.tensor(0.0, device=mag_sequence.device)
        
        # Compute energy proxy for each timestep
        energy_timeseries = []
        for t in range(time_steps):
            mag_t = mag_sequence[:, t].reshape(batch_size, channels, height, width)
            energy_t = self.compute_free_energy_proxy(mag_t)
            energy_timeseries.append(energy_t)
            
        energy_timeseries = torch.stack(energy_timeseries)
        
        # Calculate the growth rate (difference between last and first timestep)
        # Normalize by the number of timesteps
        energy_growth = (energy_timeseries[-1] - energy_timeseries[0]) / (time_steps - 1)
        
        return energy_growth
        
    def forward(self, 
                magnetogram: torch.Tensor, 
                predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-informed regularization
        
        Args:
            magnetogram: Magnetogram tensor (B, T, C, H, W)
            predictions: Dictionary of model predictions
            
        Returns:
            Regularization loss
        """
        batch_size, time_steps, channels, height, width = magnetogram.shape
        
        # Process final timestep
        mag_latest = magnetogram[:, -1, 0].unsqueeze(1)  # Use last timestep, first channel
        
        # Compute energy proxy based on spatial gradients
        magnetic_energy = self.compute_free_energy_proxy(mag_latest)
        
        # Compute magnetic complexity measure
        complexity = self.compute_magnetic_complexity(mag_latest)
        
        # Compute energy growth rate from temporal sequence
        energy_growth = self.compute_energy_growth_rate(magnetogram[:, :, 0].unsqueeze(2))
        
        # Get predicted peak flux
        if 'peak_flux_mean' in predictions:
            peak_flux = predictions['peak_flux_mean']
        else:
            peak_flux = predictions['peak_flux']
        
        # Get predicted flare probabilities
        flare_prob = predictions.get('c_vs_0', None)
        m_flare_prob = predictions.get('m_vs_0', None)
        
        # Physics-informed regularization terms
        
        # 1. Energy-flux correlation: higher energy should predict higher flux
        energy_flux_term = -torch.mean(magnetic_energy * peak_flux)
        
        # 2. Complexity-flare probability correlation
        complexity_flare_term = 0.0
        if flare_prob is not None:
            complexity_flare_term = -torch.mean(complexity * flare_prob)
        
        # 3. Energy growth rate correlation with M-class probability
        growth_m_flare_term = 0.0
        if m_flare_prob is not None:
            growth_m_flare_term = -torch.mean(energy_growth * m_flare_prob)
        
        # Combine all physics terms - weighted by importance
        physics_loss = (
            0.5 * energy_flux_term + 
            0.3 * complexity_flare_term + 
            0.2 * growth_m_flare_term
        )
        
        return self.weight * physics_loss 