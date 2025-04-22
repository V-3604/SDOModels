import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class DenseNetBackbone(nn.Module):
    """
    DenseNet backbone for feature extraction
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 pretrained: bool = True,
                 freeze_layers: bool = False):
        """
        Initialize DenseNet backbone
        
        Args:
            in_channels: Number of input channels
            pretrained: Whether to use pretrained weights
            freeze_layers: Whether to freeze backbone layers
        """
        super().__init__()
        
        # Load pretrained DenseNet121
        densenet = models.densenet121(pretrained=pretrained)
        
        # Modify first conv layer if in_channels != 3
        if in_channels != 3:
            conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
            # Initialize with pretrained weights if possible
            if pretrained and in_channels == 1:
                # For 1-channel input, use average of RGB channels
                conv1.weight.data = torch.mean(
                    densenet.features.conv0.weight.data, 
                    dim=1, 
                    keepdim=True
                )
            elif pretrained and in_channels == 4:
                # For 4-channel input, copy RGB weights and initialize alpha channel
                rgb_weight = densenet.features.conv0.weight.data
                conv1.weight.data[:, :3, :, :] = rgb_weight
                conv1.weight.data[:, 3:, :, :].normal_(0, 0.01)
                
            # Replace first conv layer
            densenet.features.conv0 = conv1
        
        # Create backbone from DenseNet features
        self.backbone = densenet.features
        
        # Feature dimensionality
        self.out_channels = 1024  # DenseNet121 final feature channels
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Features extracted by backbone
        """
        return self.backbone(x)


class MultiInputDenseNet(nn.Module):
    """
    DenseNet backbone for multiple input types (magnetogram and EUV)
    """
    def __init__(self,
                 magnetogram_channels: int = 1,
                 euv_channels: int = 7,
                 pretrained: bool = True,
                 freeze_magnetogram: bool = False,
                 freeze_euv: bool = False):
        """
        Initialize multi-input backbone
        
        Args:
            magnetogram_channels: Number of magnetogram channels
            euv_channels: Number of EUV channels
            pretrained: Whether to use pretrained weights
            freeze_magnetogram: Whether to freeze magnetogram backbone
            freeze_euv: Whether to freeze EUV backbone
        """
        super().__init__()
        
        # Create separate backbones for magnetogram and EUV images
        self.magnetogram_backbone = DenseNetBackbone(
            in_channels=magnetogram_channels,
            pretrained=pretrained,
            freeze_layers=freeze_magnetogram
        )
        
        self.euv_backbone = DenseNetBackbone(
            in_channels=euv_channels,
            pretrained=pretrained,
            freeze_layers=freeze_euv
        )
        
        # Output dimensionality
        self.out_channels = self.magnetogram_backbone.out_channels + self.euv_backbone.out_channels
        
    def forward(self, 
                magnetogram: torch.Tensor, 
                euv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multiple inputs
        
        Args:
            magnetogram: Magnetogram tensor
            euv: EUV tensor
            
        Returns:
            Tuple of magnetogram and EUV features
        """
        # Extract features from magnetogram
        magnetogram_features = self.magnetogram_backbone(magnetogram)
        
        # Extract features from EUV
        euv_features = self.euv_backbone(euv)
        
        return magnetogram_features, euv_features


class ChannelAttention(nn.Module):
    """
    Channel attention module
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize channel attention module
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for intermediate channels
        """
        super().__init__()
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Global max pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Channel attention weights
        """
        # Average pooling features
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling features
        max_out = self.fc(self.max_pool(x))
        
        # Combine features
        out = avg_out + max_out
        
        # Apply sigmoid activation
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    """
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention module
        
        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        # Ensure kernel size is odd
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        # Spatial attention convolution
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False
        )
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Spatial attention weights
        """
        # Max pooling along channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Average pooling along channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        x = torch.cat([max_out, avg_out], dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply sigmoid activation
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Initialize CBAM
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        
        # Channel attention module
        self.channel_attention = ChannelAttention(channels, reduction)
        
        # Spatial attention module
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Attended features
        """
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class MultiModalFusion(nn.Module):
    """
    Fusion module for magnetogram and EUV features
    """
    def __init__(self, 
                 magnetogram_channels: int,
                 euv_channels: int,
                 use_attention: bool = True,
                 fusion_method: str = 'concat',
                 use_normalization: bool = True,
                 dropout: float = 0.1):
        """
        Initialize fusion module
        
        Args:
            magnetogram_channels: Number of magnetogram feature channels
            euv_channels: Number of EUV feature channels
            use_attention: Whether to use attention mechanism
            fusion_method: Method to fuse features ('concat', 'sum', 'weighted_sum', 'gated', 'bilinear')
            use_normalization: Whether to use batch normalization
            dropout: Dropout rate for feature fusion
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        self.use_normalization = use_normalization
        
        # Channel normalization
        if use_normalization:
            self.magnetogram_norm = nn.BatchNorm2d(magnetogram_channels)
            self.euv_norm = nn.BatchNorm2d(euv_channels)
        
        # Apply attention to individual modalities
        if use_attention:
            self.magnetogram_attention = CBAM(magnetogram_channels)
            self.euv_attention = CBAM(euv_channels)
        else:
            self.magnetogram_attention = nn.Identity()
            self.euv_attention = nn.Identity()
            
        # Determine output channels based on fusion method
        if fusion_method == 'concat':
            self.out_channels = magnetogram_channels + euv_channels
        elif fusion_method == 'gated':
            # Gated fusion uses one modality to gate the other, then combines them
            self.mag_gate = nn.Sequential(
                nn.Conv2d(magnetogram_channels, euv_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.euv_gate = nn.Sequential(
                nn.Conv2d(euv_channels, magnetogram_channels, kernel_size=1),
                nn.Sigmoid()
            )
            # Project to common dimension
            common_channels = min(magnetogram_channels, euv_channels)
            self.mag_project = nn.Conv2d(magnetogram_channels, common_channels, kernel_size=1)
            self.euv_project = nn.Conv2d(euv_channels, common_channels, kernel_size=1)
            self.out_channels = common_channels
        elif fusion_method == 'bilinear':
            # Bilinear pooling (compact bilinear approximation)
            # Project to common dimension
            common_channels = 512
            self.mag_project = nn.Conv2d(magnetogram_channels, common_channels, kernel_size=1)
            self.euv_project = nn.Conv2d(euv_channels, common_channels, kernel_size=1)
            # Dropout for regularization
            self.dropout = nn.Dropout2d(dropout)
            self.out_channels = common_channels
        else:  # 'sum' or 'weighted_sum'
            # For sum/weighted methods, ensure channels match
            if magnetogram_channels != euv_channels:
                # Project to common dimension - use the smaller of the two
                common_channels = min(magnetogram_channels, euv_channels)
                self.mag_project = nn.Conv2d(magnetogram_channels, common_channels, kernel_size=1)
                self.euv_project = nn.Conv2d(euv_channels, common_channels, kernel_size=1)
                self.out_channels = common_channels
            else:
                self.out_channels = magnetogram_channels
                # Identity projections
                self.mag_project = nn.Identity()
                self.euv_project = nn.Identity()
            
            if fusion_method == 'weighted_sum':
                # Learnable weight parameter for weighted sum
                self.alpha = nn.Parameter(torch.tensor(0.5))
                
                # Dynamic weighting based on feature importance
                self.dynamic_weight = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.out_channels * 2, 2, kernel_size=1),
                    nn.Softmax(dim=1)
                )
                
        # Global channel attention for combined features
        if use_attention and fusion_method == 'concat':
            self.global_attention = ChannelAttention(self.out_channels)
        else:
            self.global_attention = nn.Identity()
            
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
            
    def forward(self, 
                magnetogram_features: torch.Tensor, 
                euv_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            magnetogram_features: Features from magnetogram backbone
            euv_features: Features from EUV backbone
            
        Returns:
            Fused features
        """
        # Apply normalization if enabled
        if self.use_normalization:
            magnetogram_features = self.magnetogram_norm(magnetogram_features)
            euv_features = self.euv_norm(euv_features)
            
        # Apply attention to individual modalities
        magnetogram_features = self.magnetogram_attention(magnetogram_features)
        euv_features = self.euv_attention(euv_features)
        
        # Fuse features using selected method
        if self.fusion_method == 'concat':
            fused_features = torch.cat([magnetogram_features, euv_features], dim=1)
            # Apply global attention
            fused_features = fused_features * self.global_attention(fused_features)
            
        elif self.fusion_method == 'sum':
            # Project to common dimension if needed
            mag_proj = self.mag_project(magnetogram_features)
            euv_proj = self.euv_project(euv_features)
            fused_features = mag_proj + euv_proj
            
        elif self.fusion_method == 'weighted_sum':
            # Project to common dimension if needed
            mag_proj = self.mag_project(magnetogram_features)
            euv_proj = self.euv_project(euv_features)
            
            # Dynamic weighting - compute weights based on current features
            combined = torch.cat([mag_proj.mean(dim=(2, 3), keepdim=True),
                                 euv_proj.mean(dim=(2, 3), keepdim=True)], dim=1)
            weights = self.dynamic_weight(combined)
            
            # Apply weights - more stable than using sigmoid(alpha)
            fused_features = weights[:, 0:1, :, :] * mag_proj + weights[:, 1:2, :, :] * euv_proj
            
        elif self.fusion_method == 'gated':
            # Compute gates
            mag_gated = self.mag_gate(magnetogram_features)
            euv_gated = self.euv_gate(euv_features)
            
            # Apply gates and project to common dimension
            mag_out = self.mag_project(magnetogram_features * euv_gated)
            euv_out = self.euv_project(euv_features * mag_gated)
            
            # Sum gated projections
            fused_features = mag_out + euv_out
            
        elif self.fusion_method == 'bilinear':
            # Project to common dimension
            mag_proj = self.mag_project(magnetogram_features)
            euv_proj = self.euv_project(euv_features)
            
            # Element-wise product approximating bilinear pooling
            fused_features = mag_proj * euv_proj
            
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
            
        # Apply dropout for regularization
        fused_features = self.dropout(fused_features)
            
        return fused_features 