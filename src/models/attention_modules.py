"""
Attention and Residual Modules for U-Net Improvements

This module implements the core building blocks for improved U-Net architectures:
- AttentionBlock: Attention gating mechanism for skip connections
- ResidualBlock: Residual connections for deeper networks
- AtrousConvBlock: Dilated convolutions for larger receptive fields

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


class AttentionBlock(nn.Module):
    """Attention gating mechanism for skip connections in U-Net.
    
    This block implements attention gates that allow the model to automatically
    focus on relevant features from encoder skip connections. The mechanism
    computes attention weights based on both the decoder output and encoder
    skip connection, enabling dynamic feature fusion.
    
    Algorithm:
        1. Transform decoder output (g) and encoder feature (x) to same dimension
        2. Sum the transformed features
        3. Apply ReLU activation
        4. Generate attention map (alpha) in range [0, 1] using Sigmoid
        5. Multiply encoder features by attention weights
    
    Mathematical Formulation:
        psi = sigmoid( W_psi(ReLU(W_g * g + W_x * x)) )
        output = psi * x
    
    Args:
        F_g (int): Number of feature channels from decoder (gate signal)
        F_l (int): Number of feature channels from encoder (skip connection)
        F_int (int): Number of intermediate feature channels
        
    Shape:
        Input g: [B, F_g, H_g, W_g] - decoder output
        Input x: [B, F_l, H_l, W_l] - encoder skip connection
        Output: [B, F_l, H_l, W_l] - weighted encoder features
        
    Example:
        >>> att_block = AttentionBlock(F_g=128, F_l=64, F_int=32)
        >>> decoder_feat = torch.randn(4, 128, 16, 100)  # [B, C, H, T]
        >>> encoder_feat = torch.randn(4, 64, 32, 100)
        >>> weighted_feat = att_block(decoder_feat, encoder_feat)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """Initialize the Attention Block.
        
        Args:
            F_g: Number of feature channels from decoder (gate signal)
            F_l: Number of feature channels from encoder (skip connection)
            F_int: Number of intermediate feature channels (usually F_l//2)
        """
        super(AttentionBlock, self).__init__()
        
        # Transform gate signal (decoder output)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transform skip connection (encoder features)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Generate attention map (alpha)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention block.
        
        Args:
            g: Gate signal from decoder with shape [B, F_g, H_g, W_g]
            x: Skip connection from encoder with shape [B, F_l, H_l, W_l]
            
        Returns:
            Weighted encoder features with shape [B, F_l, H_l, W_l]
        """
        # Transform gate signal to match intermediate dimension
        g1 = self.W_g(g)  # [B, F_int, H_g, W_g]
        
        # Transform encoder features to match intermediate dimension
        x1 = self.W_x(x)  # [B, F_int, H_l, W_l]
        
        # Upsample gate signal to match encoder feature size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Sum the transformed features
        psi = self.relu(g1 + x1)  # [B, F_int, H_l, W_l]
        
        # Generate attention map (alpha values in range [0, 1])
        psi = self.psi(psi)  # [B, 1, H_l, W_l]
        
        # Apply attention weights to encoder features
        return x * psi  # [B, F_l, H_l, W_l]


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers.
    
    This block implements the residual connection (shortcut) mechanism that
    helps alleviate the vanishing gradient problem in deep networks. The
    architecture consists of two consecutive convolutional layers with
    batch normalization and ReLU activation, plus a skip connection that
    adds the input to the output.
    
    Algorithm:
        1. First conv + BN + ReLU
        2. Second conv + BN
        3. Add input (residual connection)
        4. Final ReLU activation
    
    Mathematical Formulation:
        y = F(x) + x
        where F(x) = BN2(Conv2(ReLU(BN1(Conv1(x)))))
    
    Args:
        channels (int): Number of input and output channels
        
    Shape:
        Input: [B, C, H, W]
        Output: [B, C, H, W]
        
    Example:
        >>> res_block = ResidualBlock(channels=64)
        >>> x = torch.randn(4, 64, 128, 100)
        >>> out = res_block(x)
    """
    
    def __init__(self, channels: int):
        """Initialize the Residual Block.
        
        Args:
            channels: Number of input and output channels
        """
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.
        
        Args:
            x: Input tensor with shape [B, C, H, W]
            
        Returns:
            Output tensor with shape [B, C, H, W]
        """
        # Store input for residual connection
        residual = x
        
        # First conv + BN + ReLU
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv + BN
        out = self.bn2(self.conv2(out))
        
        # Add residual connection
        out += residual
        
        # Final ReLU activation
        out = self.relu(out)
        
        return out


class AtrousConvBlock(nn.Module):
    """Atrous (dilated) convolution block for multi-scale feature extraction.
    
    This block implements dilated convolutions with different dilation rates,
    which increase the receptive field without increasing the number of
    parameters. This is particularly useful for capturing long-range temporal
    dependencies in audio spectrograms.
    
    The block uses multiple parallel dilated convolutions with different
    dilation rates and concatenates their outputs for multi-scale feature
    aggregation.
    
    Algorithm:
        1. Apply parallel dilated convolutions with different dilation rates
        2. Concatenate all outputs along channel dimension
        3. Reduce dimensionality with 1x1 convolution
        
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dilation_rates (list): List of dilation rates for each branch
        kernel_size (int): Kernel size for all convolutions (default: 3)
        
    Shape:
        Input: [B, in_channels, H, W]
        Output: [B, out_channels, H, W]
        
    Example:
        >>> atrous_block = AtrousConvBlock(256, 256, dilation_rates=[2, 4, 8])
        >>> x = torch.randn(4, 256, 8, 100)
        >>> out = atrous_block(x)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: list = [2, 4, 8],
        kernel_size: int = 3
    ):
        """Initialize the Atrous Convolution Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation_rates: List of dilation rates for each branch
            kernel_size: Kernel size for all convolutions
        """
        super(AtrousConvBlock, self).__init__()
        
        # Create parallel dilated convolution branches
        # Each branch outputs out_channels // len(dilation_rates) channels
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels // len(dilation_rates),
                    kernel_size=kernel_size,
                    padding=dilation,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels // len(dilation_rates)),
                nn.ReLU(inplace=True)
            )
            for dilation in dilation_rates
        ])
        
        # Calculate actual number of channels after concatenation
        # This handles cases where out_channels is not perfectly divisible by len(dilation_rates)
        actual_concat_channels = (out_channels // len(dilation_rates)) * len(dilation_rates)
        
        # Final 1x1 convolution to reduce dimensionality
        self.final_conv = nn.Sequential(
            nn.Conv2d(actual_concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the atrous convolution block.
        
        Args:
            x: Input tensor with shape [B, in_channels, H, W]
            
        Returns:
            Output tensor with shape [B, out_channels, H, W]
        """
        # Apply all dilated convolution branches in parallel
        outputs = []
        for conv in self.dilated_convs:
            outputs.append(conv(x))
        
        # Concatenate outputs from all branches
        out = torch.cat(outputs, dim=1)
        
        # Apply final 1x1 convolution
        out = self.final_conv(out)
        
        return out


if __name__ == "__main__":
    """Test the modules with sample inputs."""
    print("Testing AttentionBlock...")
    att_block = AttentionBlock(F_g=128, F_l=64, F_int=32)
    g = torch.randn(2, 128, 16, 100)
    x = torch.randn(2, 64, 32, 100)
    out = att_block(g, x)
    print(f"  Input shapes: g={g.shape}, x={x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ AttentionBlock test passed\n")
    
    print("Testing ResidualBlock...")
    res_block = ResidualBlock(channels=64)
    x = torch.randn(2, 64, 128, 100)
    out = res_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ ResidualBlock test passed\n")
    
    print("Testing AtrousConvBlock...")
    atrous_block = AtrousConvBlock(256, 256, dilation_rates=[2, 4, 8])
    x = torch.randn(2, 256, 8, 100)
    out = atrous_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ AtrousConvBlock test passed\n")
    
    print("All tests completed successfully! ✓")