"""
Test script to verify all new models can be imported and instantiated.
"""

import torch
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
    AudioUNet5LSTM,
    AudioUNet5TemporalAttention,
    AudioUNet5ConvLSTM,
    AudioUNet5GAN,
)
from src.models import (
    SpectralLoss,
    MultiTaskLoss,
    SpectralConsistencyLoss,
    AdversarialLoss,
    Discriminator,
)

def test_model(model_class, model_name):
    """Test a single model."""
    try:
        print(f"  Testing {model_name}...")
        model = model_class().cpu()
        
        # Create dummy input
        x = torch.randn(1, 1, 256, 100)
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'generator'):
                # GAN model
                output = model.generator(x)
            else:
                output = model(x)
        
        print(f"    ✓ {model_name}: Input {x.shape} -> Output {output.shape}")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"    ✗ {model_name}: {e}")
        return False

def test_loss_function(loss_class, loss_name):
    """Test a single loss function."""
    try:
        print(f"  Testing {loss_name}...")
        loss_fn = loss_class()
        
        # Create dummy inputs
        pred = torch.randn(2, 1, 256, 100).abs()
        target = torch.randn(2, 1, 256, 100).abs()
        
        # Compute loss
        if loss_name == "AdversarialLoss":
            # Adversarial loss needs different inputs
            fake_pred = torch.randn(2, 1)
            real_pred = torch.randn(2, 1)
            g_loss = loss_fn.generator_loss(fake_pred)
            d_loss = loss_fn.discriminator_loss(real_pred, fake_pred)
            print(f"    ✓ {loss_name}: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")
        else:
            loss = loss_fn(pred, target)
            print(f"    ✓ {loss_name}: Loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"    ✗ {loss_name}: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing All Models")
    print("="*60)
    
    models = [
        (AudioUNet3, "AudioUNet3"),
        (AudioUNet5, "AudioUNet5"),
        (AudioUNet5Attention, "AudioUNet5Attention"),
        (AudioUNet5Residual, "AudioUNet5Residual"),
        (AudioUNet5Dilated, "AudioUNet5Dilated"),
        (AudioUNet5Optimized, "AudioUNet5Optimized"),
        (AudioUNet5LSTM, "AudioUNet5LSTM"),
        (AudioUNet5TemporalAttention, "AudioUNet5TemporalAttention"),
        (AudioUNet5ConvLSTM, "AudioUNet5ConvLSTM"),
        (AudioUNet5GAN, "AudioUNet5GAN"),
    ]
    
    model_results = []
    for model_class, model_name in models:
        result = test_model(model_class, model_name)
        model_results.append((model_name, result))
    
    print()
    print("="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    loss_functions = [
        (SpectralLoss, "SpectralLoss"),
        (MultiTaskLoss, "MultiTaskLoss"),
        (SpectralConsistencyLoss, "SpectralConsistencyLoss"),
        (AdversarialLoss, "AdversarialLoss"),
        (Discriminator, "Discriminator"),
    ]
    
    loss_results = []
    for loss_class, loss_name in loss_functions:
        result = test_loss_function(loss_class, loss_name)
        loss_results.append((loss_name, result))
    
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    
    print("\nModels:")
    for name, result in model_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\nLoss Functions:")
    for name, result in loss_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    # Check if all passed
    all_passed = all(r for _, r in model_results) and all(r for _, r in loss_results)
    
    print()
    if all_passed:
        print("✅ All tests passed! You can now train the new models.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")