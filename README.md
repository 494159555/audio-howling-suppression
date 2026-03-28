# Audio Howling Suppression System

<div align="center">

**A Deep Learning Approach to Audio Feedback/Howling Noise Suppression**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[中文文档](项目文档.md) | [CLAUDE Guide](CLAUDE.md) | [Documentation](docs/)

</div>

---

## 📋 Overview

This project implements an **audio howling (feedback) suppression system** using deep learning models based on U-Net architectures. The system supports 13+ U-Net variants, multiple training strategies, and traditional signal processing methods for comprehensive audio feedback elimination.

### Key Features

- 🧠 **13+ U-Net Variants**: From 3-layer baseline to GAN architectures
- 🎯 **Multiple Loss Functions**: L1, MSE, Spectral, Multi-task, Adversarial
- 🔧 **Data Augmentation**: SpecAugment, Mixup, Adversarial augmentation
- ⚡ **Training Strategies**: Mixed Precision (AMP), Curriculum Learning, Cosine Annealing
- 🎨 **Post-Processing**: Adaptive threshold, multi-frame smoothing, gain control
- 📊 **Comprehensive Evaluation**: Metrics calculation, visualization, method comparison
- 🔊 **Traditional Methods**: Frequency shift, gain suppression, adaptive feedback cancellation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GraduationProject.git
cd GraduationProject

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

Organize your audio data in the following structure:

```
data/
├── train/
│   ├── clean/
│   └── howling/
├── dev/
│   ├── clean/
│   └── howling/
└── test/
    ├── clean/
    └── howling/
```

**Audio Requirements**: WAV format, 16kHz sampling rate, paired clean/howling files with matching names.

### Training

```bash
# Train with default model (unet_v2)
python src/train.py

# Train specific model
python src/train.py --model unet_v6_optimized

# Train with YAML config (recommended)
python src/train.py --config configs/unet_v3_attention.yaml

# Override parameters
python src/train.py --config configs/unet_v2.yaml --lr 2e-4 --batch-size 4 --epochs 100
```

### Inference

```bash
# Single file inference
python scripts/inference.py \
    --model experiments/exp_xxx/checkpoints/best_model.pth \
    --input input.wav \
    --output output.wav \
    --use_griffin_lim

# Quick inference (ISTFT)
python scripts/inference.py \
    --model experiments/exp_xxx/checkpoints/best_model.pth \
    --input input.wav \
    --output output.wav \
    --device auto
```

### Evaluation

```bash
# Run tests
python tests/run_tests.py

# Compare models
python scripts/compare_models.py

# Run comprehensive evaluation
python scripts/run_experiment.py --mode comprehensive
```

---

## 📁 Project Structure

```
.
├── configs/              # YAML configuration files for all models
├── data/                 # Audio datasets (train/dev/test)
├── docs/                 # Additional documentation
├── experiments/          # Training outputs (logs, checkpoints)
├── scripts/              # Utility scripts (inference, comparison)
├── src/                  # Source code
│   ├── models/          # All U-Net variants and components
│   ├── evaluation/      # Metrics, visualization, benchmarking
│   ├── traditional/     # Traditional signal processing methods
│   ├── config.py        # Global configuration
│   ├── dataset.py       # Data loading and preprocessing
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
└── tests/               # Automated test suite
```

---

## 🎯 Available Models

### Baseline Models
| Model | Description | Use Case |
|-------|-------------|----------|
| `unet_v1` | 3-layer U-Net | Fast prototyping |
| `unet_v2` | 5-layer U-Net | General purpose (default) |

### Enhanced Models
| Model | Description | Features |
|-------|-------------|----------|
| `unet_v3_attention` | Attention mechanism | Focus on important frequencies |
| `unet_v4_residual` | Residual connections | Deeper networks |
| `unet_v5_dilated` | Dilated convolutions | Large receptive field |
| `unet_v6_optimized` | Combined optimization | Attention + Residual + Dilated (recommended) |

### Temporal Models
| Model | Description | Use Case |
|-------|-------------|----------|
| `unet_v7_lstm` | Bidirectional LSTM | Time-varying howling |
| `unet_v8_temporal_attention` | Temporal attention | Focus on specific time periods |
| `unet_v9_convlstm` | ConvLSTM | Spatiotemporal modeling |

### Advanced Models
| Model | Description | Features |
|-------|-------------|----------|
| `unet_v10_gan` | GAN architecture | Best generation quality |
| `unet_v11_multiscale` | Multi-scale features | Multi-scale feature extraction |
| `unet_v12_pyramid` | Pyramid pooling | Global context |
| `unet_v13_fpn` | Feature pyramid | Strong semantics + fine details |

### Training Strategy Configs
| Config | Description | Use Case |
|--------|-------------|----------|
| `unet_v14_mixed_precision` | Mixed precision training | GPU acceleration |
| `unet_v15_curriculum` | Curriculum learning | Stable convergence |
| `unet_v16_lr_scheduler` | Advanced LR scheduling | Optimized convergence |

---

## 🔧 Configuration

### Priority
```
CLI Arguments > YAML Config > src/config.py defaults
```

### Example Config

```yaml
# configs/unet_v2.yaml
model:
  name: unet_v2
  class: AudioUNet5

training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 50
  loss_function: l1

loss:
  type: l1

data_augmentation:
  enabled: false

training_strategies:
  mixed_precision: false
  lr_scheduler: plateau
```

---

## 📊 Training & Monitoring

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir experiments/
# Visit http://localhost:6006
```

### Experiment Output

```
experiments/exp_YYYYMMDD_HHMMSS_model_name/
├── checkpoints/
│   └── best_model.pth      # Best validation loss checkpoint
├── logs/                    # TensorBoard logs
├── config_backup.py         # Configuration snapshot
└── config.json              # JSON config
```

---

## 📖 Documentation

- [中文文档](项目文档.md) - Comprehensive Chinese documentation
- [CLAUDE.md](CLAUDE.md) - Claude Code usage guide
- [Scripts Guide](scripts/README.md) - Utility scripts documentation
- [Tests Guide](tests/README.md) - Testing suite documentation
- [Configs Guide](configs/README.md) - Configuration files documentation

---

## 🛠️ Development

### Running Tests

```bash
# Quick test
python tests/run_tests.py

# Full test
python tests/run_tests.py --mode full

# Model test
python tests/run_tests.py --mode models
```

### Adding New Models

1. Create model file in `src/models/unet_vXX_name.py`
2. Import in `src/models/__init__.py`
3. Register in `src/config.py` `AVAILABLE_MODELS`
4. (Optional) Create YAML config in `configs/`
5. Test with `python scripts/test_models.py`

---

## 📝 Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `batch_size` (8→4→2) or use smaller model |
| Loss not decreasing | Check learning rate, data pairing, TensorBoard |
| Poor inference quality | Try `--use_griffin_lim`, ensure 16kHz input, train longer |
| Can't find model | Check registration in `src/config.py` |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- U-Net architecture inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Audio processing techniques from [librosa](https://librosa.org/)
- PyTorch framework from [PyTorch Team](https://pytorch.org/)

---

<div align="center">

**Made with ❤️ for Audio Signal Processing**

</div>
