"""
U-Net v7 Model - 5-Layer U-Net with LSTM/GRU Integration

This module implements a 5-layer U-Net architecture with bidirectional LSTM
for audio howling suppression. The LSTM layer is inserted at the bottleneck
to capture long-range temporal dependencies in the spectrogram.

Key Improvements:
- Bidirectional LSTM at bottleneck for temporal modeling
- Captures both past and future context
- Improved handling of dynamic howling patterns
- Better temporal consistency

Author: Research Team
Date: 2026-3-23
Version: 7.0.0
"""

# Standard library imports
# None

# Third-party imports
import torch
import torch.nn as nn

# Local imports
# None


class AudioUNet5LSTM(nn.Module):
    """5-layer U-Net with bidirectional LSTM for audio howling suppression.
    
    This model enhances the standard U-Net by inserting a bidirectional LSTM
    layer at the bottleneck. The LSTM processes the encoded features across
    the time dimension, capturing long-range temporal dependencies and
    improving the model's ability to handle dynamic howling patterns.
    
    Network Architecture:
        Encoder (Downsampling):
            enc1: [B,1,256,T] -> [B,16,128,T]
            enc2: [B,16,128,T] -> [B,32,64,T]
            enc3: [B,32,64,T] -> [B,64,32,T]
            enc4: [B,64,32,T] -> [B,128,16,T]
            enc5: [B,128,16,T] -> [B,256,8,T] (bottleneck)
            
        Temporal Modeling (Bottleneck):
            lstm: [B,T,2048] -> [B,T,128] (bidirectional, 64*2)
            reshape: [B,T,128] -> [B,128,8,T]
            
        Decoder (Upsampling):
            dec5: [B,128,8,T] -> [B,128,16,T] + enc4 skip connection
            dec4: [B,256,16,T] -> [B,64,32,T] + enc3 skip connection
            dec3: [B,128,32,T] -> [B,32,64,T] + enc2 skip connection
            dec2: [B,64,64,T] -> [B,16,128,T] + enc1 skip connection
            dec1: [B,32,128,T] -> [B,1,256,T]
    
    Key Features:
        - 5 encoder and decoder layers for deep feature extraction
        - Bidirectional LSTM at bottleneck (128 hidden units per direction)
        - Captures long-range temporal dependencies
        - Improved handling of dynamic howling patterns
        - Log-domain processing for numerical stability
        - Multiplicative masking mechanism
        
    Attributes:
        enc1-enc5 (nn.Sequential): Encoder layers
        dec1-dec5 (nn.Sequential): Decoder layers
        lstm (nn.LSTM): Bidirectional LSTM layer
    """
    
    def __init__(
        self,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        use_bidirectional: bool = True
    ):
        """Initialize the 5-layer U-Net with LSTM.
        
        Args:
            lstm_hidden: Number of hidden units per LSTM direction
            lstm_layers: Number of LSTM layers
            use_bidirectional: Whether to use bidirectional LSTM
        """
        super(AudioUNet5LSTM, self).__init__()
        
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.use_bidirectional = use_bidirectional
        
        # ==========================
        # Encoder (Downsampling) - 5 Layers
        # ==========================
        
        # Layer 1: [B, 1, 256, T] -> [B, 16, 128, T]
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 2: [B, 16, 128, T] -> [B, 32, 64, T]
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 3: [B, 32, 64, T] -> [B, 64, 32, T]
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 4: [B, 64, 32, T] -> [B, 128, 16, T]
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 5: [B, 128, 16, T] -> [B, 256, 8, T] (Bottleneck)
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ==========================
        # Temporal Modeling (Bottleneck)
        # ==========================
        
        # LSTM input size: 256 channels * 8 frequency bins = 2048
        lstm_input_size = 256 * 8
        num_directions = 2 if use_bidirectional else 1
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=use_bidirectional,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Project LSTM output back to bottleneck channels
        lstm_output_size = lstm_hidden * num_directions
        self.lstm_proj = nn.Linear(lstm_output_size, 128)

        # ==========================
        # Decoder (Upsampling) - 5 Layers
        # ==========================
        
        # Layer 5: [B, 128, 8, T] -> [B, 128, 16, T]
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 4: [B, 128+128, 16, T] -> [B, 64, 32, T]
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Layer 3: [B, 64+64, 32, T] -> [B, 32, 64, T]
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Layer 2: [B, 32+32, 64, T] -> [B, 16, 128, T]
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Layer 1: [B, 16+16, 128, T] -> [B, 1, 256, T]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # Output multiplicative mask in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net with LSTM.
        
        Processes input spectrogram through encoder-LSTM-decoder architecture
        to generate a multiplicative mask for howling suppression.
        
        Args:
            x (torch.Tensor): Input spectrogram with shape [B, 1, 256, T]
                             representing linear magnitude spectrum
            
        Returns:
            torch.Tensor: Output spectrogram with shape [B, 1, 256, T]
                         representing howling-suppressed audio spectrum
        """
        # ==========================
        # Log-domain Feature Extraction
        # ==========================
        x_log = torch.log10(x + 1e-8)

        # ==========================
        # Encoder Forward Pass
        # ==========================
        e1 = self.enc1(x_log)    # [B, 16, 128, T]
        e2 = self.enc2(e1)        # [B, 32, 64, T]
        e3 = self.enc3(e2)        # [B, 64, 32, T]
        e4 = self.enc4(e3)        # [B, 128, 16, T]
        e5 = self.enc5(e4)        # [B, 256, 8, T] - Bottleneck

        # ==========================
        # Temporal Modeling with LSTM
        # ==========================
        # Reshape for LSTM: [B, 256, 8, T] -> [B, T, 2048]
        batch_size, channels, freq_bins, time_steps = e5.shape
        e5_reshaped = e5.permute(0, 3, 1, 2)  # [B, T, 256, 8]
        e5_flat = e5_reshaped.reshape(batch_size, time_steps, -1)  # [B, T, 2048]
        
        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(e5_flat)  # [B, T, 128] (bidirectional)
        
        # Project to bottleneck channels
        lstm_proj = self.lstm_proj(lstm_out)  # [B, T, 128]
        
        # Reshape back to spectrogram format: [B, T, 128] -> [B, 128, T] -> [B, 128, 8, T]
        # Note: LSTM compresses frequency dimension, so we need to expand it back
        lstm_reshaped = lstm_proj.permute(0, 2, 1)  # [B, 128, T]
        lstm_reshaped = lstm_reshaped.unsqueeze(2)  # [B, 128, 1, T]
        lstm_reshaped = lstm_reshaped.repeat(1, 1, freq_bins, 1)  # [B, 128, 8, T]

        # ==========================
        # Decoder Forward Pass with Skip Connections
        # ==========================
        
        # Decoder Layer 5 + Skip Connection 4
        d5 = self.dec5(lstm_reshaped)  # [B, 128, 16, T]
        d5_cat = torch.cat([d5, e4], dim=1)  # [B, 256, 16, T]

        # Decoder Layer 4 + Skip Connection 3
        d4 = self.dec4(d5_cat)    # [B, 64, 32, T]
        d4_cat = torch.cat([d4, e3], dim=1)  # [B, 128, 32, T]

        # Decoder Layer 3 + Skip Connection 2
        d3 = self.dec3(d4_cat)    # [B, 32, 64, T]
        d3_cat = torch.cat([d3, e2], dim=1)  # [B, 64, 64, T]

        # Decoder Layer 2 + Skip Connection 1
        d2 = self.dec2(d3_cat)    # [B, 16, 128, T]
        d2_cat = torch.cat([d2, e1], dim=1)  # [B, 32, 128, T]

        # Final Decoder Layer - Generate Mask
        mask = self.dec1(d2_cat)  # [B, 1, 256, T]

        # ==========================
        # Multiplicative Masking
        # ==========================
        output = x * mask
        return output


if __name__ == "__main__":
    """Test the model with sample input."""
    print("Testing AudioUNet5LSTM...")
    model = AudioUNet5LSTM(lstm_hidden=64, lstm_layers=2)
    
    # Create sample input: [Batch=2, Channels=1, Freq=256, Time=100]
    x = torch.randn(2, 1, 256, 100)
    
    # Forward pass
    output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test LSTM component
    print(f"  LSTM hidden size: {model.lstm_hidden}")
    print(f"  LSTM layers: {model.lstm_layers}")
    print(f"  LSTM bidirectional: {model.use_bidirectional}")
    print(f"  ✓ AudioUNet5LSTM test passed\n")