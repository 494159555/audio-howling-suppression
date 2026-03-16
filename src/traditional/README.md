# 传统音频啸叫抑制方法

三种传统音频啸叫抑制算法，用于与深度学习方法对比。

## 文件结构

```
src/traditional/
├── __init__.py              # 模块初始化
├── frequency_shift.py       # 移频移向法
├── gain_suppression.py      # 增益抑制法
├── adaptive_feedback.py     # 自适应反馈抵消法
├── test.py                  # 简单测试脚本
└── README.md               # 说明文档
```

## 方法介绍

### 1. 移频移向法 (FrequencyShiftMethod)
- **原理**: 轻微频率偏移破坏反馈相位条件
- **参数**: `shift_hz` (默认20Hz)
- **特点**: 音质影响小，适合轻微啸叫

### 2. 增益抑制法 (GainSuppressionMethod)  
- **原理**: 检测并抑制啸叫频段
- **参数**: `threshold_db` (默认-30dB)
- **特点**: 针对性强，适合明显啸叫

### 3. 自适应反馈抵消法 (AdaptiveFeedbackMethod)
- **原理**: 自适应滤波估计反馈路径
- **参数**: `filter_length` (默认64), `step_size` (默认0.01)
- **特点**: 自适应强，适合复杂环境

## 使用方法

```python
from src.traditional import (
    FrequencyShiftMethod,
    GainSuppressionMethod, 
    AdaptiveFeedbackMethod
)

# 创建方法实例
freq_method = FrequencyShiftMethod(shift_hz=20.0)
gain_method = GainSuppressionMethod(threshold_db=-30.0)
adaptive_method = AdaptiveFeedbackMethod(filter_length=64)

# 处理音频频谱数据 (log域)
# input_shape: [batch_size, channels, freq_bins, time_frames]
output = freq_method(input_data)
```

## 运行测试

```bash
python src/traditional/test.py
```

## 输入输出格式

- **输入**: `[B, 1, F, T]` log域幅度谱
- **输出**: `[B, 1, F, T]` log域幅度谱

## 性能对比

| 方法 | 啸叫抑制 | 音质保持 | 计算复杂度 | 适用场景 |
|------|----------|----------|------------|----------|
| 移频移向法 | 轻微 | 优秀 | 低 | 轻微啸叫 |
| 增益抑制法 | 明显 | 中等 | 中 | 明显啸叫 |
| 自适应反馈抵消法 | 中等 | 良好 | 高 | 复杂环境 |
