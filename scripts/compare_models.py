"""
Model Comparison Script

This script compares all U-Net models (baseline and improved versions) by:
1. Initializing each model
2. Testing forward pass with sample input
3. Counting parameters
4. Displaying comparison results

Usage:
    python scripts/compare_models.py

Author: Research Team
Date: 2026-3-23
Version: 1.0.0
"""

# Standard library imports
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
)
from src.config import Config, cfg


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary containing parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
    }


def format_number(num: int) -> str:
    """Format large numbers with comma separators.
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def test_model(model_class, model_name: str, input_tensor: torch.Tensor) -> dict:
    """Test a model with sample input.
    
    Args:
        model_class: Model class to test
        model_name: Name of the model
        input_tensor: Input tensor for testing
        
    Returns:
        dict: Test results including shapes and parameter counts
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize model
        model = model_class()
        
        # Move to device
        model = model.to(cfg.DEVICE)
        input_tensor = input_tensor.to(cfg.DEVICE)
        
        # Count parameters
        params = count_parameters(model)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Collect results
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': tuple(output.shape),
            'params': params,
            'success': True,
            'error': None,
        }
        
        # Print results
        print(f"✓ Model initialized successfully")
        print(f"  Input shape:  {results['input_shape']}")
        print(f"  Output shape: {results['output_shape']}")
        print(f"  Total parameters:       {format_number(params['total']):>10} ({params['total']:,})")
        print(f"  Trainable parameters:    {format_number(params['trainable']):>10} ({params['trainable']:,})")
        print(f"  Non-trainable params:   {format_number(params['non_trainable']):>10} ({params['non_trainable']:,})")
        print(f"✓ Forward pass successful")
        
    except Exception as e:
        results = {
            'name': model_name,
            'input_shape': tuple(input_tensor.shape),
            'output_shape': None,
            'params': None,
            'success': False,
            'error': str(e),
        }
        print(f"✗ Model failed with error:")
        print(f"  {e}")
    
    return results


def print_summary_table(results: list, config: Config):
    """Print a summary comparison table of all models.
    
    Args:
        results: List of test results for each model
        config: Configuration object
    """
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE - Model Comparison")
    print(f"{'='*80}\n")
    
    # Print header
    header = f"{'Model':<25} {'Total Params':>15} {'Trainable':>15} {'Status':>15}"
    print(header)
    print("-" * 80)
    
    # Print each model
    for result in results:
        if result['success']:
            params_str = format_number(result['params']['total'])
            trainable_str = format_number(result['params']['trainable'])
            status = "✓ PASS"
        else:
            params_str = "N/A"
            trainable_str = "N/A"
            status = "✗ FAIL"
        
        row = f"{result['name']:<25} {params_str:>15} {trainable_str:>15} {status:>15}"
        print(row)
    
    # Print footer
    print("-" * 80)
    
    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    if successful > 0:
        total_params = [r['params']['total'] for r in results if r['success']]
        min_params = min(total_params)
        max_params = max(total_params)
        avg_params = sum(total_params) / len(total_params)
        
        print(f"\nStatistics:")
        print(f"  Successful models: {successful}/{total}")
        print(f"  Min parameters:  {format_number(min_params):>10} ({min_params:,})")
        print(f"  Max parameters:  {format_number(max_params):>10} ({max_params:,})")
        print(f"  Avg parameters:  {format_number(int(avg_params)):>10} ({int(avg_params):,})")
    
    # Print model descriptions
    print(f"\n\n{'='*80}")
    print("MODEL DESCRIPTIONS")
    print(f"{'='*80}\n")
    
    for model_key, description in config.MODEL_DESCRIPTIONS.items():
        print(f"  {model_key:<20} : {description}")


def main():
    """Main function to run model comparison."""
    print("\n" + "="*80)
    print("U-Net Model Comparison Script")
    print("="*80)
    print(f"\nDevice: {cfg.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Define models to test
    models_to_test = [
        (AudioUNet3, 'AudioUNet3 (3-layer baseline)'),
        (AudioUNet5, 'AudioUNet5 (5-layer baseline)'),
        (AudioUNet5Attention, 'AudioUNet5Attention (Attention mechanism)'),
        (AudioUNet5Residual, 'AudioUNet5Residual (Residual connections)'),
        (AudioUNet5Dilated, 'AudioUNet5Dilated (Atrous convolutions)'),
        (AudioUNet5Optimized, 'AudioUNet5Optimized (All improvements)'),
    ]
    
    # Create sample input
    # Shape: [Batch=2, Channels=1, Freq=256, Time=100]
    sample_input = torch.randn(2, 1, 256, 100)
    print(f"\nSample input shape: {tuple(sample_input.shape)}")
    
    # Test all models
    results = []
    for model_class, model_name in models_to_test:
        result = test_model(model_class, model_name, sample_input)
        results.append(result)
    
    # Print summary table
    print_summary_table(results, cfg)
    
    # Print conclusion
    print(f"\n\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results if r['success'])
    if successful == len(results):
        print("✓ All models initialized and tested successfully!")
        print("\nYou can now use any of these models for training:")
        for model_key, description in cfg.AVAILABLE_MODELS.items():
            print(f"  - {model_key}: {description}")
    else:
        print(f"✗ {len(results) - successful} model(s) failed to initialize")
        print("\nPlease check the error messages above for details.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()