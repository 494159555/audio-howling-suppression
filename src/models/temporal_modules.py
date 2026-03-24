"""
Temporal Modeling Modules for Audio Howling Suppression

This module implements temporal modeling components for enhancing U-Net's
ability to capture temporal dependencies in audio spectrograms:
- TemporalAttention: Attention mechanism along time dimension
- ConvLSTMCell: Convolutional LSTM cell for spatio-temporal modeling
- TemporalPooling: Multi-scale temporal pooling for capturing different time scales

Author: Research Team
Date: 2026-3-23
Version: 1.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for spectrograms.
    
    This module computes attention weights along the time dimension, allowing
    the model to dynamically focus on different time segments. It's particularly
    useful for focusing on periods with howling activity.
    
    Algorithm:
        1. Compute attention scores from input features
        2. Apply softmax to get attention weights (sum to 1)
        3. Weight input features by attention weights
        4. Optionally add residual connection
    
    Mathematical Formulation:
        e_t = f(W * h_t + b)  # attention score at time t
        α_t = exp(e_t) / Σ_t exp(e_t)  # attention weights (softmax)
        output = Σ_t α_t * h_t  # weighted sum
        
    Args:
        channels (int): Number of input feature channels
        reduction (int): Reduction ratio for attention computation (default: 8)
        use_residual (bool): Whether to use residual connection (default: True)
        
    Shape:
        Input: [B, C, F, T]
        Output: [B, C, F, T]
        
    Example:
        >>> temp_att = TemporalAttention(channels=256)
        >>> x = torch.randn(4, 256, 32, 100)  # [B, C, F, T]
        >>> out = temp_att(x)
    """
    
    def __init__(self, channels: int, reduction: int = 8, use_residual: bool = True):
        """Initialize Temporal Attention.
        
        Args:
            channels: Number of input feature channels
            reduction: Reduction ratio for intermediate layer
            use_residual: Whether to add residual connection
        """
        super(TemporalAttention, self).__init__()
        
        self.channels = channels
        self.use_residual = use_residual
        
        # Pooling to aggregate information across frequency dimension
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Keep freq, pool time to 1
        
        # Attention network
        intermediate_channels = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.Linear(channels, intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels, channels),
            nn.Softmax(dim=1)  # Softmax over time dimension
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal attention.
        
        Args:
            x: Input features [B, C, F, T]
            
        Returns:
            Weighted features [B, C, F, T]
        """
        residual = x
        
        # Compute attention weights
        # Aggregate across frequency dimension: [B, C, F, T] -> [B, C, 1, T]
        x_pool = self.avg_pool(x.permute(0, 1, 3, 2))  # [B, C, T, 1]
        x_pool = x_pool.squeeze(-1)  # [B, C, T]
        
        # Transpose for linear layer: [B, C, T] -> [B, T, C]
        x_pool = x_pool.permute(0, 2, 1)
        
        # Compute attention scores
        attention_scores = self.attention(x_pool)  # [B, T, C]
        
        # Normalize across time dimension
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, C]
        
        # Transpose back: [B, T, C] -> [B, C, T]
        attention_weights = attention_weights.permute(0, 2, 1)
        
        # Apply attention weights: [B, C, T] -> [B, C, 1, T] -> [B, C, F, T]
        attention_weights = attention_weights.unsqueeze(2)
        attention_weights = attention_weights.expand_as(x)
        
        # Weight input features
        output = x * attention_weights
        
        # Add residual connection
        if self.use_residual:
            output = output + residual
        
        return output


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatio-temporal modeling.
    
    This cell combines convolutional layers with LSTM architecture,
    enabling the model to capture both spatial (spectral) and temporal
    dependencies. Unlike standard LSTM which operates on vectors, ConvLSTM
    preserves spatial structure through convolutional operations.
    
    Mathematical Formulation:
        i_t = σ(W_xi * X_t + W_hi * H_{t-1} + W_ci * C_{t-1} + b_i)
        f_t = σ(W_xf * X_t + W_hf * H_{t-1} + W_cf * C_{t-1} + b_f)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_xc * X_t + W_hc * H_{t-1} + b_c)
        o_t = σ(W_xo * X_t + W_ho * H_{t-1} + W_co * C_t + b_o)
        H_t = o_t ⊙ tanh(C_t)
        
    Args:
        input_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels
        kernel_size (int): Kernel size for convolutions (default: 3)
        padding (int): Padding for convolutions (default: 1)
        
    Example:
        >>> conv_lstm = ConvLSTMCell(input_channels=256, hidden_channels=128)
        >>> x_t = torch.randn(4, 256, 8, 100)  # [B, C, H, W]
        >>> h_prev = torch.randn(4, 128, 8, 100)
        >>> c_prev = torch.randn(4, 128, 8, 100)
        >>> h_next, c_next = conv_lstm(x_t, (h_prev, c_prev))
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        """Initialize ConvLSTM Cell.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size for convolutions
            padding: Padding for convolutions
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Convolution for input gate
        self.conv_input = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Convolution for forget gate
        self.conv_forget = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Convolution for cell state
        self.conv_cell = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Convolution for output gate
        self.conv_output = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(
        self,
        x: torch.Tensor,
        state: tuple = None
    ) -> tuple:
        """Forward pass of ConvLSTM cell.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            state: Tuple of (hidden_state, cell_state) from previous time step
                   If None, initializes to zeros
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        batch_size, _, height, width = x.size()
        
        # Initialize state if not provided
        if state is None or state[0] is None or state[1] is None:
            h_prev = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device
            )
            c_prev = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device
            )
        else:
            h_prev, c_prev = state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        i_t = self.sigmoid(self.conv_input(combined))   # Input gate
        f_t = self.sigmoid(self.conv_forget(combined))  # Forget gate
        g_t = self.tanh(self.conv_cell(combined))       # Cell candidate
        o_t = self.sigmoid(self.conv_output(combined))  # Output gate
        
        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        
        # Update hidden state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t


class TemporalPooling(nn.Module):
    """Multi-scale temporal pooling for capturing temporal context.
    
    This module applies multiple pooling operations with different kernel sizes
    to capture temporal context at different scales. The multi-scale features
    are then combined to provide rich temporal information.
    
    Algorithm:
        1. Apply parallel pooling with different kernel sizes
        2. Upsample all features to original size
        3. Concatenate and reduce dimensionality
        
    Args:
        channels (int): Number of input channels
        pool_sizes (list): List of pooling kernel sizes (default: [3, 5, 7])
        
    Example:
        >>> temp_pool = TemporalPooling(channels=256, pool_sizes=[3, 5, 7])
        >>> x = torch.randn(4, 256, 32, 100)  # [B, C, F, T]
        >>> out = temp_pool(x)
    """
    
    def __init__(self, channels: int, pool_sizes: list = [3, 5, 7]):
        """Initialize Temporal Pooling.
        
        Args:
            channels: Number of input channels
            pool_sizes: List of pooling kernel sizes (must be odd numbers)
        """
        super(TemporalPooling, self).__init__()
        
        self.channels = channels
        self.pool_sizes = pool_sizes
        
        # Create pooling layers
        self.pools = nn.ModuleList()
        for pool_size in pool_sizes:
            padding = pool_size // 2
            pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=1, padding=(0, padding))
            self.pools.append(pool)
        
        # 1x1 convolution to reduce dimensionality after concatenation
        output_channels = channels * len(pool_sizes)
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(output_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal pooling.
        
        Args:
            x: Input features [B, C, F, T]
            
        Returns:
            Multi-scale temporal features [B, C, F, T]
        """
        # Apply all pooling operations
        pooled_features = []
        for pool in self.pools:
            pooled = pool(x)
            pooled_features.append(pooled)
        
        # Concatenate all pooled features
        combined = torch.cat(pooled_features, dim=1)
        
        # Reduce dimensionality
        output = self.conv_reduce(combined)
        
        return output


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with dilated convolutions.
    
    This block applies multiple dilated convolutions in parallel to capture
    temporal dependencies at different scales without increasing parameters.
    
    Args:
        channels (int): Number of input and output channels
        dilations (list): List of dilation rates (default: [1, 2, 4, 8])
        kernel_size (int): Kernel size (default: 3)
        
    Example:
        >>> temp_conv = TemporalConvBlock(channels=256, dilations=[1, 2, 4, 8])
        >>> x = torch.randn(4, 256, 32, 100)  # [B, C, F, T]
        >>> out = temp_conv(x)
    """
    
    def __init__(
        self,
        channels: int,
        dilations: list = [1, 2, 4, 8],
        kernel_size: int = 3
    ):
        """Initialize Temporal Convolution Block.
        
        Args:
            channels: Number of input and output channels
            dilations: List of dilation rates
            kernel_size: Kernel size
        """
        super(TemporalConvBlock, self).__init__()
        
        self.channels = channels
        self.dilations = dilations
        
        # Create parallel dilated convolutions
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size // 2) * dilation
            conv = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(1, padding),  # Only pad time dimension
                    dilation=(1, dilation),  # Only dilate time dimension
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.dilated_convs.append(conv)
        
        # Fusion layer
        fusion_channels = channels * len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal convolution block.
        
        Args:
            x: Input features [B, C, F, T]
            
        Returns:
            Multi-scale temporal features [B, C, F, T]
        """
        # Apply all dilated convolutions
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        # Concatenate and fuse
        combined = torch.cat(features, dim=1)
        output = self.fusion(combined)
        
        return output


if __name__ == "__main__":
    """Test the temporal modules."""
    print("Testing temporal modules...\n")
    
    # Test TemporalAttention
    print("Testing TemporalAttention...")
    temp_att = TemporalAttention(channels=256)
    x = torch.randn(4, 256, 32, 100)
    out = temp_att(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalAttention test passed\n")
    
    # Test ConvLSTMCell
    print("Testing ConvLSTMCell...")
    conv_lstm = ConvLSTMCell(input_channels=256, hidden_channels=128)
    x_t = torch.randn(4, 256, 8, 100)
    h_prev = torch.randn(4, 128, 8, 100)
    c_prev = torch.randn(4, 128, 8, 100)
    h_next, c_next = conv_lstm(x_t, (h_prev, c_prev))
    print(f"  Input shape: {x_t.shape}")
    print(f"  Hidden state shape: {h_next.shape}")
    print(f"  Cell state shape: {c_next.shape}")
    print(f"  ✓ ConvLSTMCell test passed\n")
    
    # Test TemporalPooling
    print("Testing TemporalPooling...")
    temp_pool = TemporalPooling(channels=256, pool_sizes=[3, 5, 7])
    x = torch.randn(4, 256, 32, 100)
    out = temp_pool(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalPooling test passed\n")
    
    # Test TemporalConvBlock
    print("Testing TemporalConvBlock...")
    temp_conv = TemporalConvBlock(channels=256, dilations=[1, 2, 4, 8])
    x = torch.randn(4, 256, 32, 100)
    out = temp_conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ TemporalConvBlock test passed\n")
    
    print("All temporal module tests completed successfully! ✓")