import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal modeling of feature sequences
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 recurrent_dropout: float = 0.2):
        """
        Initialize temporal LSTM
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            recurrent_dropout: Dropout probability for recurrent connections
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
            recurrent_dropout=recurrent_dropout
        )
        
        # Output size accounting for bidirectionality
        self.out_channels = hidden_size * 2 if bidirectional else hidden_size
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Tuple of output features and hidden states
        """
        return self.lstm(x)


class TemporalGRU(nn.Module):
    """
    Bidirectional GRU for temporal modeling of feature sequences
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 recurrent_dropout: float = 0.2):
        """
        Initialize temporal GRU
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            recurrent_dropout: Dropout probability for recurrent connections
        """
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
            recurrent_dropout=recurrent_dropout
        )
        
        # Output size accounting for bidirectionality
        self.out_channels = hidden_size * 2 if bidirectional else hidden_size
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Tuple of output features and hidden states
        """
        return self.gru(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models
    """
    def __init__(self, d_model: int, max_seq_length: int = 10, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Feature dimensionality
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head self-attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV projections combined for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
            
        Returns:
            Self-attended tensor
        """
        batch_size, seq_length, _ = x.shape
        
        # Project inputs to queries, keys, and values
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_length, head_dim]
        
        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer
    """
    def __init__(self, 
                 d_model: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        Initialize transformer encoder layer
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        # Multi-head self-attention with residual connection and normalization
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attn(x))
        
        # Feedforward with residual connection and normalization
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal modeling
    """
    def __init__(self, 
                 input_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 10):
        """
        Initialize temporal transformer
        
        Args:
            input_size: Size of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Input projection if needed
        if input_size != d_model:
            self.input_proj = nn.Linear(input_size, d_model)
        else:
            self.input_proj = nn.Identity()
            
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output size
        self.out_channels = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Transformed tensor
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return x


class SpatioTemporalFusion(nn.Module):
    """
    Fusion module for spatial and temporal features
    """
    def __init__(self, 
                 spatial_channels: int,
                 temporal_type: str = 'lstm',
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 transformer_nhead: int = 8,
                 transformer_dim_feedforward: int = 2048,
                 max_seq_length: int = 8):
        """
        Initialize spatiotemporal fusion module
        
        Args:
            spatial_channels: Number of spatial feature channels
            temporal_type: Type of temporal model ('lstm', 'gru', 'transformer')
            hidden_size: Size of hidden state for RNN or transformer d_model
            num_layers: Number of layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            transformer_nhead: Number of attention heads (for transformer)
            transformer_dim_feedforward: Dimension of feedforward network (for transformer)
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.temporal_type = temporal_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Add feature normalization
        self.feature_norm = nn.LayerNorm(spatial_channels)
        
        # Create temporal model based on type
        if temporal_type == 'lstm':
            # Add dropout specifically for recurrent connections (different from regular dropout)
            self.temporal_model = TemporalLSTM(
                input_size=spatial_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                recurrent_dropout=0.2  # Add recurrent dropout
            )
            # Add temporal attention for focusing on key timesteps
            self.temporal_attention = TemporalAttention(
                hidden_size * (2 if bidirectional else 1),
                hidden_size * (2 if bidirectional else 1)
            )
        elif temporal_type == 'gru':
            self.temporal_model = TemporalGRU(
                input_size=spatial_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                recurrent_dropout=0.2  # Add recurrent dropout
            )
            # Add temporal attention for focusing on key timesteps
            self.temporal_attention = TemporalAttention(
                hidden_size * (2 if bidirectional else 1),
                hidden_size * (2 if bidirectional else 1)
            )
        elif temporal_type == 'transformer':
            self.temporal_model = TemporalTransformer(
                input_size=spatial_channels,
                d_model=hidden_size,
                nhead=transformer_nhead,
                num_layers=num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                max_seq_length=max_seq_length
            )
            # For transformer, we'll use weighted pooling over temporal dimension
            self.temporal_attention = TemporalWeightedPooling(hidden_size)
        else:
            raise ValueError(f"Unsupported temporal model type: {temporal_type}")
            
        # Output size
        if temporal_type in ['lstm', 'gru']:
            self.out_channels = hidden_size * (2 if bidirectional else 1)
        else:  # transformer
            self.out_channels = hidden_size
            
        # Add final layer normalization
        self.output_norm = nn.LayerNorm(self.out_channels)
        
        # Time-dependent feature gating
        self.feature_gate = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.Sigmoid()
        )
    
    def handle_variable_sequence(self, x, lengths=None):
        """
        Handle variable sequence lengths by masking or packing
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Processed tensor and lengths
        """
        batch_size, time_steps, features = x.shape
        
        if lengths is None:
            # Infer sequence lengths by finding timesteps with non-zero feature sums
            # This assumes empty/padding timesteps have all zeros
            feature_sums = torch.sum(torch.abs(x), dim=2)  # Sum across feature dimension
            mask = feature_sums > 0
            lengths = torch.sum(mask, dim=1).clamp(min=1)  # Ensure at least length 1
        
        return x, lengths
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, channels, height, width) for CNN features
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Temporally modeled features
        """
        batch_size, time_steps, channels, height, width = x.shape
        
        # Reshape for temporal modeling
        # Convert from [B, T, C, H, W] to [B, T, C*H*W]
        x_flat = x.reshape(batch_size, time_steps, -1)
        
        # Apply feature normalization
        x_flat = self.feature_norm(x_flat)
        
        # Handle variable sequence lengths
        x_flat, lengths = self.handle_variable_sequence(x_flat, lengths)
        
        # Apply temporal model
        if self.temporal_type == 'lstm':
            # Pack padded sequence if we have lengths
            if lengths is not None:
                lengths_cpu = lengths.cpu()
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x_flat, lengths_cpu.clamp(min=1), 
                    batch_first=True, enforce_sorted=False
                )
                # Pass through LSTM
                packed_output, (hidden, cell) = self.temporal_model(x_packed)
                # Unpack output
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                output, (hidden, cell) = self.temporal_model(x_flat)
            
            # Apply temporal attention to focus on key timesteps
            attended_output = self.temporal_attention(output, lengths)
            
            # Gate the features to focus on relevant dimensions
            gates = self.feature_gate(attended_output)
            gated_output = attended_output * gates
            
            # Apply final normalization
            final_output = self.output_norm(gated_output)
            
            return final_output
        
        elif self.temporal_type == 'gru':
            # Pack padded sequence if we have lengths
            if lengths is not None:
                lengths_cpu = lengths.cpu()
                x_packed = nn.utils.rnn.pack_padded_sequence(
                    x_flat, lengths_cpu.clamp(min=1), 
                    batch_first=True, enforce_sorted=False
                )
                # Pass through GRU
                packed_output, hidden = self.temporal_model(x_packed)
                # Unpack output
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                output, hidden = self.temporal_model(x_flat)
            
            # Apply temporal attention to focus on key timesteps
            attended_output = self.temporal_attention(output, lengths)
            
            # Gate the features to focus on relevant dimensions
            gates = self.feature_gate(attended_output)
            gated_output = attended_output * gates
            
            # Apply final normalization
            final_output = self.output_norm(gated_output)
            
            return final_output
        
        elif self.temporal_type == 'transformer':
            # Create padding mask if we have lengths
            mask = None
            if lengths is not None:
                # Create attention mask
                mask = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=x.device)
                for i, length in enumerate(lengths):
                    mask[i, length:] = True
            
            # Apply transformer
            output = self.temporal_model(x_flat, src_key_padding_mask=mask)
            
            # Apply weighted pooling over temporal dimension
            final_output = self.temporal_attention(output, lengths)
            
            # Apply final normalization
            final_output = self.output_norm(final_output)
            
            return final_output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important timesteps
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, sequence, lengths=None):
        """
        Apply attention over temporal dimension
        
        Args:
            sequence: Sequence tensor of shape (batch_size, time_steps, features)
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Context vector of shape (batch_size, features)
        """
        # Calculate attention scores
        scores = self.attention(sequence)  # (batch_size, time_steps, 1)
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = torch.arange(sequence.size(1), device=sequence.device).expand(
                lengths.size(0), sequence.size(1)
            ) >= lengths.unsqueeze(1)
            scores.masked_fill_(mask.unsqueeze(2), -1e9)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights to sequence
        context = torch.sum(weights * sequence, dim=1)  # (batch_size, features)
        
        return context


class TemporalWeightedPooling(nn.Module):
    """
    Learnable weighted pooling over temporal dimension
    """
    def __init__(self, input_size):
        super().__init__()
        
        self.weight_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1)
        )
        
    def forward(self, sequence, lengths=None):
        """
        Apply weighted pooling over temporal dimension
        
        Args:
            sequence: Sequence tensor of shape (batch_size, time_steps, features)
            lengths: Optional tensor of sequence lengths
            
        Returns:
            Pooled features of shape (batch_size, features)
        """
        # Calculate importance weights for each timestep
        weights = self.weight_net(sequence)  # (batch_size, time_steps, 1)
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = torch.arange(sequence.size(1), device=sequence.device).expand(
                lengths.size(0), sequence.size(1)
            ) >= lengths.unsqueeze(1)
            weights.masked_fill_(mask.unsqueeze(2), -1e9)
        
        # Apply softmax to get normalized weights
        norm_weights = F.softmax(weights, dim=1)  # (batch_size, time_steps, 1)
        
        # Apply weighted pooling
        pooled = torch.sum(norm_weights * sequence, dim=1)  # (batch_size, features)
        
        return pooled 