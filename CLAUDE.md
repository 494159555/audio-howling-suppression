# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供项目指导。

## 项目概述

这是一个**音频啸叫抑制系统**，使用深度学习模型（主要为U-Net变体）和传统信号处理方法来消除音频反馈/啸叫噪声。项目支持13+种U-Net变体架构、多种训练策略和评估方法。

### 核心功能
- **13+种U-Net变体**：从3层基线到GAN架构
- **CNN/RNN基线模型**：用于对比评估
- **多种损失函数**：L1、MSE、频谱、多任务、对抗损失
- **数据增强策略**：音频增强、SpecAugment、Mixup、对抗增强
- **训练策略**：混合精度(AMP)、课程学习、余弦退火预热、单周期策略
- **后处理优化**：自适应阈值、多帧平滑、增益控制
- **传统方法**：移频移向、增益抑制、自适应反馈抑制
- **完整评估系统**：指标计算、可视化、方法对比、基准测试

---

## 开发命令

### 训练模型

```bash
# 使用默认配置训练（默认使用unet_v2）
python src/train.py

# 训练指定模型
python src/train.py --model unet_v6_optimized

# 使用YAML配置文件训练（推荐）
python src/train.py --config configs/unet_v3_attention.yaml

# 覆盖配置参数
python src/train.py --config configs/unet_v2.yaml --lr 2e-4 --batch-size 4 --epochs 100

# 自定义实验名称
python src/train.py --model unet_v6_optimized --exp-name "experiment_name"
```

### 运行测试

```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定模型测试
python test_models.py
```

### 评估模型

```bash
# 评估训练好的模型
python src/evaluate.py --checkpoint experiments/exp_xxx/checkpoints/best_model.pth

# 快速评估传统方法
python -c "from src.evaluation.test_runner import run_quick_evaluation; run_quick_evaluation(methods=['frequency_shift', 'gain_suppression'], num_samples=5)"
```

### 推理

```bash
# 单文件推理（使用Griffin-Lim，音质更好）
python inference.py --model experiments/exp_xxx/checkpoints/best_model.pth --input input.wav --output output.wav --use_griffin_lim

# 快速推理（使用ISTFT）
python inference.py --model experiments/exp_xxx/checkpoints/best_model.pth --input input.wav --output output.wav --device auto
```

### 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir experiments/
# 访问 http://localhost:6006
```

---

## 项目架构

### 模型变体（13种U-Net）

| 版本 | 类名 | 特点 |
|-----|------|------|
| **v1** | AudioUNet3 | 3层基线U-Net |
| **v2** | AudioUNet5 | 5层基线U-Net（默认模型） |
| **v3** | AudioUNet5Attention | 添加注意力门机制 |
| **v4** | AudioUNet5Residual | 添加残差连接 |
| **v5** | AudioUNet5Dilated | 添加空洞卷积 |
| **v6** | AudioUNet5Optimized | 组合优化：注意力+残差+空洞 |
| **v7** | AudioUNet5LSTM | 添加双向LSTM时序建模 |
| **v8** | AudioUNet5TemporalAttention | 添加时间注意力机制 |
| **v9** | AudioUNet5ConvLSTM | 添加ConvLSTM层 |
| **v10** | AudioUNet5GAN | GAN架构（含判别器） |
| **v11** | AudioUNet5MultiScale | 多尺度特征提取（3+5+7层） |
| **v12** | AudioUNet5Pyramid | 金字塔池化模块 |
| **v13** | AudioUNet5FPN | 特征金字塔网络 |

**基线模型**：
- `AudioCNN`（位于 `src/models/CNN.py`）：4层卷积自编码器
- `AudioRNN`（位于 `src/models/RNN.py`）：CNN + 双向LSTM混合架构

### 配置系统

项目采用**三层配置系统**：

1. **全局配置**（`src/config.py`）：定义所有默认参数、模型映射和路径
2. **YAML配置**（`configs/*.yaml`）：各模型专属配置文件
3. **命令行参数**：优先级最高，可覆盖前两者

配置优先级：`CLI参数 > YAML配置 > src/config.py默认值`

### 关键组件

- **`src/config.py`**：全局配置类 `Config`，包含所有默认值。使用 `@property` 实现 `DEVICE` 自动检测
- **`src/dataset.py`**：`HowlingDataset` 类，加载成对音频（clean/howling），应用STFT、对数变换、归一化。支持数据增强
- **`src/train.py`**：主训练脚本，支持实验管理、TensorBoard日志、检查点保存
- **`src/models/`**：所有模型实现、损失函数、数据增强、训练策略、后处理
- **`src/evaluation/`**：指标计算、可视化、方法对比、基准测试
- **`src/traditional/`**：频率移位、增益抑制、自适应反馈等传统方法

### 数据处理流程

```
原始音频对 (clean, howling)
    ↓ 长度归一化（填充/截断到3秒）
    ↓ STFT变换（512窗口，128跳跃）
    ↓ 取幅度谱 → 对数变换
    ↓ 归一化到 [0, 1]
    ↓ 裁剪频率维度到 256
输出: [batch, channel, 256, time]
```

音频参数：采样率16kHz、3秒片段、512 FFT窗口、128跳跃长度

### 训练功能

**损失函数**（位于 `src/models/loss_functions.py`）：
- `L1Loss`：L1距离损失
- `MSELoss`：均方误差损失
- `SpectralLoss`：对数域频谱损失
- `SpectralConsistencyLoss`：频谱一致性损失（平滑性约束）
- `MultiTaskLoss`：多任务组合损失
- `AdversarialLoss`：GAN对抗损失（含判别器）

**数据增强**（位于 `src/models/augmentation.py`）：
- `AudioAugmentation`：音频级增强（噪声、音调、位移）
- `SpecAugment`：频谱图掩蔽（频率/时间掩蔽）
- `MixupAugmentation`：样本混合增强
- `AdversarialAugmentation`：对抗性增强
- `CombinedAugmentation`：组合增强策略

**训练策略**（位于 `src/models/training_strategies.py`）：
- `MixedPrecisionTrainer`：混合精度训练（AMP）
- `CosineAnnealingWarmupScheduler`：余弦退火+预热调度器
- `OneCycleScheduler`：单周期学习率策略
- `CurriculumLearning`：课程学习（渐进难度）

**学习率调度器类型**：
- `plateau`：ReduceLROnPlateau
- `cosine_warmup`：余弦退火+预热
- `one_cycle`：单周期策略
- `step`：步进衰减

**后处理**（位于 `src/models/post_processing.py`）：
- `AdaptivePostProcessing`：自适应阈值处理
- `MultiFrameSmoother`：多帧平滑（移动平均/Kalman/维纳/中值）
- `AdaptiveGainControl`：自适应增益控制（AGC/DRC/限幅器）
- `PostProcessingPipeline`：组合后处理流水线

### 实验目录结构

```
experiments/exp_YYYYMMDD_HHMMSS_model_name/
├── checkpoints/
│   └── best_model.pth      # 验证集Loss最佳的检查点
├── logs/                    # TensorBoard日志
├── config_backup.py         # 配置快照
└── config.json              # JSON格式配置（train_v2.py）
```

---

## 模型注册

模型在 `src/config.py` 中注册：

```python
AVAILABLE_MODELS = {
    'unet_v1': 'AudioUNet3',
    'unet_v2': 'AudioUNet5',
    # ...
}
```

添加新模型步骤：
1. 在 `src/models/` 创建模型文件（如 `unet_vXX_name.py`）
2. 在 `src/models/__init__.py` 导入
3. 在 `src/config.py` 的 `AVAILABLE_MODELS` 字典中添加映射
4. 在 `MODEL_DESCRIPTIONS` 添加描述
5. （可选）在 `configs/` 创建专属YAML配置

---

## 重要文件位置

- **根配置文件**：`configs/` - 各模型的YAML配置
- **全局配置**：`src/config.py` - 所有默认值和模型注册表
- **模型定义**：`src/models/unet_*.py`
- **训练入口**：`src/train.py`
- **数据集**：`src/dataset.py`
- **测试脚本**：`tests/run_tests.py`、根目录的 `test_*.py` 文件
- **文档**：根目录的中文文档（`项目文档.md`、`重构方案文档.md` 等）

---

## 开发注意事项

- **设备选择**：使用 `cfg.DEVICE` 属性（自动检测CUDA）
- **批大小**：默认8，如OOM可降至4或2
- **音频格式**：WAV文件，推荐16kHz采样率
- **数据结构**：`data/{train,dev,test}/{clean,howling}/` - 成对文件名必须匹配
- **模型输出**：应用于输入频谱的乘性掩码 [0,1]
- **相位重构**：使用Griffin-Lim获得更好音质，使用ISTFT获得更快速度

---

## 常见问题

| 问题 | 解决方案 |
|-----|---------|
| CUDA显存不足(OOM) | 减小 `batch_size`（8→4→2）或使用更小模型（unet_v1） |
| Loss不下降 | 检查学习率、验证数据配对、查看TensorBoard |
| 推理音质差 | 尝试 `--use_griffin_lim`、确保16kHz输入、增加训练轮数 |
| 配置错误 | YAML配置只覆盖指定键，其他使用 `src/config.py` 默认值 |
| 找不到模型 | 确认已在 `src/config.py` 的 `AVAILABLE_MODELS` 中注册 |

---

## 配置文件示例

```yaml
# configs/unet_v3_attention.yaml
model:
  name: unet_v3_attention
  class: AudioUNet5Attention

training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 50
  loss_function: multitask

loss:
  type: multitask
  weights:
    spectral: 0.5
    l1: 0.3
    mse: 0.2

training_strategies:
  mixed_precision: true
  lr_scheduler: cosine_warmup
  warmup_epochs: 5

data_augmentation:
  enabled: true
  spec_augment:
    freq_masking: true
    time_masking: true
```

---

## 模型选择建议

| 场景 | 推荐模型 |
|-----|---------|
| 快速原型验证 | unet_v1 |
| 标准使用 | unet_v2 |
| 追求性能 | unet_v6_optimized |
| 时序问题严重 | unet_v7_lstm / unet_v8_temporal_attention |
| 复杂声学场景 | unet_v10_gan |
| 需要多尺度特征 | unet_v11_multiscale / unet_v13_fpn |
