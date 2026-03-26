# Scripts 目录说明

本目录包含项目开发中常用的辅助脚本和工具。

## 📁 脚本列表

### 1. inference.py - 音频推理脚本
**功能**: 使用训练好的模型对音频进行啸叫抑制处理

**使用方法**:
```bash
# 基本用法
python scripts/inference.py --model <模型路径> --input <输入音频> --output <输出音频>

# 使用GPU加速
python scripts/inference.py --model exp/best.pth --input noisy.wav --output clean.wav --device cuda

# 使用高质量相位重构
python scripts/inference.py --model exp/best.pth --input noisy.wav --output clean.wav --use_griffin_lim
```

**主要参数**:
- `--model`: 模型检查点路径（.pth文件）
- `--input`: 输入音频文件（.wav文件）
- `--output`: 输出音频文件（.wav文件）
- `--device`: 计算设备（auto/cpu/cuda）
- `--use_griffin_lim`: 使用Griffin-Lim算法重构相位（默认启用）

**适用场景**:
- 训练完成后对音频进行推理
- 批量处理音频文件
- 对比不同模型的推理效果

---

### 2. test_models.py - 模型测试脚本
**功能**: 验证所有模型是否可以正确导入和初始化

**使用方法**:
```bash
python scripts/test_models.py
```

**输出内容**:
- 每个模型的初始化状态
- 模型参数量统计
- 前向传播测试结果
- 输入输出形状信息

**适用场景**:
- 添加新模型后验证集成
- 环境迁移后检查模型可用性
- 快速查看模型参数量对比

---

### 3. compare_models.py - 模型对比脚本
**功能**: 全面对比所有U-Net模型变体

**使用方法**:
```bash
python scripts/compare_models.py
```

**输出内容**:
- 详细的参数量对比表
- 模型测试结果汇总
- 参数统计信息（最少/最多/平均）
- 模型选择建议
- 模型详细描述

**适用场景**:
- 选择最适合的模型
- 评估模型计算复杂度
- 对比不同变体的改进效果
- 性能分析和优化参考

---

### 4. run_experiment.py - 实验运行脚本
**功能**: 运行科学评估实验，对比不同方法

**使用方法**:
```bash
# 快速评估（默认10个样本）
python scripts/run_experiment.py --mode quick

# 全面评估（整个测试集）
python scripts/run_experiment.py --mode comprehensive --batch_size 8

# 仅评估传统方法
python scripts/run_experiment.py --mode traditional

# 自定义方法组合
python scripts/run_experiment.py --mode custom --methods unet_v2 frequency_shift
```

**实验模式**:
- `quick`: 快速评估少量样本
- `comprehensive`: 全面评估整个测试集
- `traditional`: 仅评估传统方法
- `custom`: 自定义方法组合

**适用场景**:
- 方法性能对比
- 生成实验报告
- 评估模型在实际数据上的表现

---

### 5. update_model_comments.py - 注释更新工具
**功能**: 批量更新模型文件的注释和文档字符串

**使用方法**:
```bash
# 预览将要更新的内容
python scripts/update_model_comments.py --all --preview

# 更新所有模型文件并备份
python scripts/update_model_comments.py --all --backup

# 更新指定模型
python scripts/update_model_comments.py --model unet_v2
```

**主要参数**:
- `--all`: 更新所有模型文件
- `--model`: 更新指定的模型文件
- `--preview`: 预览模式（不实际修改）
- `--backup`: 创建备份文件

**适用场景**:
- 统一代码注释风格
- 添加缺失的文档字符串
- 更新模型描述信息

---

## 📊 脚本使用流程

### 训练后的完整工作流

```bash
# 1. 训练模型
python src/train.py --model unet_v2 --exp-name "my_experiment"

# 2. 测试所有模型是否可用
python scripts/test_models.py

# 3. 对比模型参数和性能
python scripts/compare_models.py

# 4. 对音频进行推理
python scripts/inference.py \
    --model experiments/exp_XXX_my_experiment/checkpoints/best_model.pth \
    --input data/test/noisy/sample_001.wav \
    --output results/output.wav \
    --device cuda

# 5. 运行全面评估
python scripts/run_experiment.py --mode comprehensive
```

---

## 💡 使用建议

### 开发阶段
1. 使用 `test_models.py` 快速验证模型集成
2. 使用 `compare_models.py` 选择合适的模型
3. 使用 `inference.py` 测试单文件推理效果

### 测试阶段
1. 使用 `run_experiment.py --mode quick` 快速评估
2. 对比不同方法的效果
3. 分析评估结果和可视化

### 生产阶段
1. 使用 `run_experiment.py --mode comprehensive` 全面评估
2. 生成详细的实验报告
3. 使用 `inference.py` 批量处理音频

---

## 🔧 脚本开发规范

### 新增脚本时请遵循：

1. **文件命名**: 使用小写字母和下划线（如 `my_script.py`）
2. **文档字符串**: 添加详细的模块文档字符串
3. **中文注释**: 代码注释使用中文，方便理解
4. **命令行参数**: 使用 `argparse` 提供清晰的参数说明
5. **错误处理**: 添加异常捕获和友好的错误提示
6. **输出格式**: 使用清晰的输出格式（表格、进度条等）

### 文档字符串模板：

```python
#!/usr/bin/env python3
"""脚本简短描述

详细描述脚本的功能、用途和注意事项。

主要功能：
    - 功能1
    - 功能2

使用方法：
    python scripts/script_name.py 参数

作者：XXX
版本：1.0
"""
```

---

## 📝 维护说明

### 脚本更新记录

| 脚本 | 最后更新 | 主要变更 |
|------|---------|---------|
| inference.py | 2025-03-26 | 添加详细中文注释和文档 |
| test_models.py | 2025-03-26 | 添加详细中文注释和文档 |
| compare_models.py | 2025-03-26 | 添加详细中文注释和文档 |
| run_experiment.py | 2025-03-26 | 添加详细中文注释和文档 |
| update_model_comments.py | 2025-03-26 | 新增工具 |

---

## 🆘 常见问题

### Q: 脚本运行报错找不到模块？
**A**: 确保在项目根目录运行，或检查 Python 路径设置。

### Q: inference.py 输出音质不好？
**A**:
1. 尝试使用 `--use_griffin_lim` 参数
2. 检查输入音频采样率（建议16kHz）
3. 确认模型训练是否充分

### Q: test_models.py 显示模型测试失败？
**A**:
1. 检查模型代码是否有语法错误
2. 确认模型已正确导入到 `src/models/__init__.py`
3. 确认模型已在 `src/config.py` 中注册

---

## 📞 获取帮助

如需更多帮助，请查看：
- 项目主文档: `项目文档.md`
- CLAUDE.md: Claude Code 使用指南
- 各脚本的帮助命令: `python scripts/<script_name>.py --help`
