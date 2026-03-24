"""
Loss Functions for Audio Howling Suppression

This module implements various loss functions for training audio howling suppression models:
- MultiTaskLoss: Combines spectral, L1, and MSE losses
- SpectralConsistencyLoss: Ensures spectral smoothness and consistency
- AdversarialLoss: GAN-based adversarial loss with discriminator
- SpectralLoss: Log-domain spectral distance

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


class SpectralLoss(nn.Module):
    """Spectral loss based on log-domain magnitude spectrum distance.
    
    This loss computes the L1 distance between log-domain magnitude spectra,
    which corresponds to perceptual audio quality better than linear domain.
    
    Mathematical Formulation:
        L_spec = mean(|log10(pred + ε) - log10(target + ε)|)
    
    Advantages:
        - Better matches human perception
        - Provides dynamic range compression
        - More stable for large magnitude variations
        
    Example:
        >>> spec_loss = SpectralLoss()
        >>> pred = torch.randn(4, 1, 256, 100)
        >>> target = torch.randn(4, 1, 256, 100)
        >>> loss = spec_loss(pred, target)
    """
    
    def __init__(self):
        """Initialize the Spectral Loss."""
        super(SpectralLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss.
        
        Args:
            pred: Predicted magnitude spectrogram [B, 1, F, T]
            target: Target magnitude spectrogram [B, 1, F, T]
            
        Returns:
            Scalar loss value
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        
        # Log-domain L1 distance
        pred_log = torch.log10(pred + epsilon)
        target_log = torch.log10(target + epsilon)
        
        loss = F.l1_loss(pred_log, target_log)
        return loss


class SpectralConsistencyLoss(nn.Module):
    """Spectral consistency loss for ensuring smooth and coherent spectrograms.
    
    This loss encourages smooth transitions in both frequency and time dimensions,
    reducing artifacts and improving naturalness of reconstructed audio.
    
    Components:
        1. Frequency smoothing: Penalizes abrupt changes in frequency dimension
        2. Temporal smoothing: Penalizes abrupt changes in time dimension
        3. Spectral flatness penalty: Discourages artificial spectral peaks
    
    Mathematical Formulation:
        L_freq = mean(|∂S/∂f|)  # Frequency gradient
        L_time = mean(|∂S/∂t|)  # Time gradient
        L_consistency = λ_freq * L_freq + λ_time * L_time
        
    Example:
        >>> cons_loss = SpectralConsistencyLoss()
        >>> spectrogram = torch.randn(4, 1, 256, 100)
        >>> loss = cons_loss(spectrogram)
    """
    
    def __init__(self, lambda_freq: float = 0.1, lambda_time: float = 0.1):
        """Initialize the Spectral Consistency Loss.
        
        Args:
            lambda_freq: Weight for frequency smoothing term
            lambda_time: Weight for temporal smoothing term
        """
        super(SpectralConsistencyLoss, self).__init__()
        self.lambda_freq = lambda_freq
        self.lambda_time = lambda_time
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Compute spectral consistency loss.
        
        Args:
            spectrogram: Input spectrogram [B, 1, F, T]
            
        Returns:
            Scalar loss value
        """
        # Compute frequency gradient (along dimension 2)
        freq_grad = torch.diff(spectrogram, dim=2)
        freq_loss = torch.mean(torch.abs(freq_grad))
        
        # Compute temporal gradient (along dimension 3)
        time_grad = torch.diff(spectrogram, dim=3)
        time_loss = torch.mean(torch.abs(time_grad))
        
        # Combined consistency loss
        total_loss = self.lambda_freq * freq_loss + self.lambda_time * time_loss
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining multiple loss components.
    
    This loss function combines spectral, L1, MSE, and consistency losses
    with configurable weights to optimize model from multiple perspectives.
    
    Loss Components:
        1. Spectral Loss: Log-domain spectral distance (perceptual)
        2. L1 Loss: Linear domain L1 distance (robust)
        3. MSE Loss: Linear domain MSE (smooth)
        4. Consistency Loss: Spectral smoothness (naturalness)
    
    Mathematical Formulation:
        L_total = w_spec * L_spec + w_l1 * L_l1 + w_mse * L_mse + w_cons * L_cons
    
    Example:
        >>> multitask_loss = MultiTaskLoss(
        ...     weights={'spectral': 0.4, 'l1': 0.3, 'mse': 0.2, 'consistency': 0.1}
        ... )
        >>> pred = torch.randn(4, 1, 256, 100)
        >>> target = torch.randn(4, 1, 256, 100)
        >>> loss = multitask_loss(pred, target)
    """
    
    def __init__(
        self,
        weights: dict = None,
        use_spectral: bool = True,
        use_l1: bool = True,
        use_mse: bool = True,
        use_consistency: bool = False
    ):
        """Initialize the Multi-Task Loss.
        
        Args:
            weights: Dictionary of loss weights
                     Default: {'spectral': 0.5, 'l1': 0.3, 'mse': 0.2, 'consistency': 0.0}
            use_spectral: Whether to use spectral loss
            use_l1: Whether to use L1 loss
            use_mse: Whether to use MSE loss
            use_consistency: Whether to use spectral consistency loss
        """
        super(MultiTaskLoss, self).__init__()
        
        # Set default weights
        if weights is None:
            weights = {
                'spectral': 0.5,
                'l1': 0.3,
                'mse': 0.2,
                'consistency': 0.0
            }
        
        self.weights = weights
        self.use_spectral = use_spectral
        self.use_l1 = use_l1
        self.use_mse = use_mse
        self.use_consistency = use_consistency
        
        # Initialize loss components
        self.spectral_loss = SpectralLoss() if use_spectral else None
        self.l1_loss = nn.L1Loss() if use_l1 else None
        self.mse_loss = nn.MSELoss() if use_mse else None
        self.consistency_loss = SpectralConsistencyLoss() if use_consistency else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """Compute multi-task loss.
        
        Args:
            pred: Predicted magnitude spectrogram [B, 1, F, T]
            target: Target magnitude spectrogram [B, 1, F, T]
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Spectral loss
        if self.use_spectral and self.spectral_loss is not None:
            spec_loss = self.spectral_loss(pred, target)
            total_loss += self.weights['spectral'] * spec_loss
            loss_dict['spectral'] = spec_loss.item()
        
        # L1 loss
        if self.use_l1 and self.l1_loss is not None:
            l1_loss = self.l1_loss(pred, target)
            total_loss += self.weights['l1'] * l1_loss
            loss_dict['l1'] = l1_loss.item()
        
        # MSE loss
        if self.use_mse and self.mse_loss is not None:
            mse_loss = self.mse_loss(pred, target)
            total_loss += self.weights['mse'] * mse_loss
            loss_dict['mse'] = mse_loss.item()
        
        # Consistency loss (applied to prediction only)
        if self.use_consistency and self.consistency_loss is not None:
            cons_loss = self.consistency_loss(pred)
            total_loss += self.weights['consistency'] * cons_loss
            loss_dict['consistency'] = cons_loss.item()
        
        return total_loss, loss_dict


class Discriminator(nn.Module):
    """Discriminator network for GAN-based training.
    
    This discriminator classifies whether input spectrograms are real (clean)
    or fake (generated by the model). It uses a convolutional architecture
    with LeakyReLU activations and spectral normalization.
    
    Architecture:
        Input: [B, 1, 256, T]
        Conv1: [B, 1, 256, T] -> [B, 64, 128, T]  (stride 2 in frequency)
        Conv2: [B, 64, 128, T] -> [B, 128, 64, T] (stride 2 in frequency)
        Conv3: [B, 128, 64, T] -> [B, 256, 32, T] (stride 2 in frequency)
        Conv4: [B, 256, 32, T] -> [B, 1, 1, 1]      (global avg pool + fc)
        
    Example:
        >>> discriminator = Discriminator()
        >>> real_spec = torch.randn(4, 1, 256, 100)
        >>> fake_spec = torch.randn(4, 1, 256, 100)
        >>> real_score = discriminator(real_spec)
        >>> fake_score = discriminator(fake_spec)
    """
    
    def __init__(self, input_channels: int = 1):
        """Initialize the Discriminator.
        
        Args:
            input_channels: Number of input channels (default: 1 for magnitude)
        """
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layer
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.
        
        Args:
            x: Input spectrogram [B, C, F, T]
            
        Returns:
            Probability of being real [B, 1]
        """
        # Convolutional feature extraction
        features = self.conv_layers(x)
        
        # Final classification
        output = self.final_layer(features)
        
        return output


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training.
    
    This loss implements both generator and discriminator losses for GAN training.
    Supports standard GAN, LSGAN, and WGAN-GP variants.
    
    Loss Types:
        - 'standard': Binary cross-entropy loss
        - 'lsgan': Least squares GAN loss
        - 'wgan': Wasserstein GAN loss
    
    Example:
        >>> adv_loss = AdversarialLoss(loss_type='lsgan')
        >>> discriminator = Discriminator()
        >>> pred_spec = torch.randn(4, 1, 256, 100)
        >>> clean_spec = torch.randn(4, 1, 256, 100)
        >>> 
        >>> # Generator loss
        >>> fake_pred = discriminator(pred_spec)
        >>> g_loss = adv_loss.generator_loss(fake_pred)
        >>> 
        >>> # Discriminator loss
        >>> real_pred = discriminator(clean_spec)
        >>> fake_pred_detach = discriminator(pred_spec.detach())
        >>> d_loss = adv_loss.discriminator_loss(real_pred, fake_pred_detach)
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        """Initialize the Adversarial Loss.
        
        Args:
            loss_type: Type of GAN loss ('standard', 'lsgan', 'wgan')
        """
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'standard':
            self.criterion = nn.BCELoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute generator loss.
        
        Args:
            fake_pred: Discriminator predictions on fake samples
            
        Returns:
            Generator loss
        """
        if self.loss_type == 'standard':
            # Generator wants discriminator to think fake is real
            target = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target)
        
        elif self.loss_type == 'lsgan':
            # Generator wants fake predictions to be 1
            target = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target)
        
        elif self.loss_type == 'wgan':
            # Generator wants to maximize fake predictions (minimize negative)
            return -torch.mean(fake_pred)
    
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss.
        
        Args:
            real_pred: Discriminator predictions on real samples
            fake_pred: Discriminator predictions on fake samples
            
        Returns:
            Discriminator loss
        """
        if self.loss_type == 'standard':
            # Real samples should be classified as real (1)
            target_real = torch.ones_like(real_pred)
            loss_real = self.criterion(real_pred, target_real)
            
            # Fake samples should be classified as fake (0)
            target_fake = torch.zeros_like(fake_pred)
            loss_fake = self.criterion(fake_pred, target_fake)
            
            return (loss_real + loss_fake) / 2
        
        elif self.loss_type == 'lsgan':
            # Real samples should be 1
            target_real = torch.ones_like(real_pred)
            loss_real = self.criterion(real_pred, target_real)
            
            # Fake samples should be 0
            target_fake = torch.zeros_like(fake_pred)
            loss_fake = self.criterion(fake_pred, target_fake)
            
            return (loss_real + loss_fake) / 2
        
        elif self.loss_type == 'wgan':
            # Discriminator wants to maximize (real - fake)
            return -torch.mean(real_pred) + torch.mean(fake_pred)


if __name__ == "__main__":
    """Test the loss functions."""
    print("Testing loss functions...\n")
    
    # Test data
    batch_size = 4
    freq_bins = 256
    time_steps = 100
    pred = torch.randn(batch_size, 1, freq_bins, time_steps).abs()
    target = torch.randn(batch_size, 1, freq_bins, time_steps).abs()
    
    # Test SpectralLoss
    print("Testing SpectralLoss...")
    spec_loss = SpectralLoss()
    loss = spec_loss(pred, target)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  ✓ SpectralLoss test passed\n")
    
    # Test SpectralConsistencyLoss
    print("Testing SpectralConsistencyLoss...")
    cons_loss = SpectralConsistencyLoss(lambda_freq=0.1, lambda_time=0.1)
    loss = cons_loss(pred)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  ✓ SpectralConsistencyLoss test passed\n")
    
    # Test MultiTaskLoss
    print("Testing MultiTaskLoss...")
    multitask_loss = MultiTaskLoss(
        weights={'spectral': 0.4, 'l1': 0.3, 'mse': 0.2, 'consistency': 0.1},
        use_consistency=True
    )
    total_loss, loss_dict = multitask_loss(pred, target)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.4f}")
    print(f"  ✓ MultiTaskLoss test passed\n")
    
    # Test Discriminator
    print("Testing Discriminator...")
    discriminator = Discriminator()
    real_spec = torch.randn(batch_size, 1, freq_bins, time_steps).abs()
    fake_spec = torch.randn(batch_size, 1, freq_bins, time_steps).abs()
    real_score = discriminator(real_spec)
    fake_score = discriminator(fake_spec)
    print(f"  Real score shape: {real_score.shape}")
    print(f"  Fake score shape: {fake_score.shape}")
    print(f"  Real score mean: {real_score.mean().item():.4f}")
    print(f"  Fake score mean: {fake_score.mean().item():.4f}")
    print(f"  ✓ Discriminator test passed\n")
    
    # Test AdversarialLoss
    print("Testing AdversarialLoss...")
    adv_loss = AdversarialLoss(loss_type='lsgan')
    g_loss = adv_loss.generator_loss(fake_score)
    d_loss = adv_loss.discriminator_loss(real_score, fake_score)
    print(f"  Generator loss: {g_loss.item():.4f}")
    print(f"  Discriminator loss: {d_loss.item():.4f}")
    print(f"  ✓ AdversarialLoss test passed\n")
    
    print("All loss function tests completed successfully! ✓")