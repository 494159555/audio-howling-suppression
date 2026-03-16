# 音频啸叫抑制方法科学评估系统

## 概述

本评估系统提供了全面的音频啸叫抑制方法性能评估功能，支持多种传统方法和深度学习方法的对比分析。

## 功能特性

### 🎯 核心功能
- **多方法支持**: 支持移频移向法、增益抑制法、自适应反馈抵消法、UNet深度学习方法
- **全面指标**: SNR改善、PSNR、STOI、啸叫抑制效果、计算效率等
- **可视化报告**: 频谱图对比、雷达图、柱状图、综合报告
- **统计分析**: 方法排名、显著性检验、推荐建议

### 📊 评估指标
- **音频质量指标**:
  - SNR改善 (dB): 信噪比改善程度
  - PSNR (dB): 峰值信噪比
  - STOI (0-1): 短时客观可懂度
  - MOS分数 (1-5): 主观音质评分估算

- **啸叫抑制指标**:
  - 啸叫衰减率 (dB): 高频段能量衰减
  - 频谱平滑度改善: 频谱连续性改善

- **计算效率指标**:
  - 处理时间 (ms): 单样本处理延迟
  - 内存使用 (MB): 运行时内存占用
  - 参数量: 模型复杂度

## 快速开始

### 1. 基础使用

```python
from src.evaluation.test_runner import run_quick_evaluation

# 快速评估传统方法（5个样本）
results = run_quick_evaluation(
    methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
    num_samples=5
)
```

### 2. 全面评估

```python
from src.evaluation.test_runner import evaluate_all_methods

# 评估所有可用方法
results = evaluate_all_methods()
```

### 3. 自定义评估

```python
from src.evaluation.test_runner import run_comprehensive_evaluation

# 自定义评估配置
results = run_comprehensive_evaluation(
    methods=['frequency_shift', 'unet'],  # 指定方法
    batch_size=4,                          # 批大小
    save_results=True,                      # 保存结果
    generate_visualizations=True             # 生成可视化
)
```

## 详细使用指南

### 评估指标计算

```python
from src.evaluation.metrics import AudioMetrics, calculate_mos_score

# 创建指标计算器
metrics_calculator = AudioMetrics()

# 计算各项指标
snr_improvement = metrics_calculator.calculate_snr(clean, enhanced, noisy)
psnr = metrics_calculator.calculate_psnr(clean, enhanced)
stoi = metrics_calculator.calculate_stoi(clean, enhanced)

# 计算MOS分数
metrics = {
    'snr_improvement_db': snr_improvement,
    'psnr_db': psnr,
    'stoi_score': stoi,
    'howling_reduction_db': 5.0
}
mos_score = calculate_mos_score(metrics)
```

### 可视化生成

```python
from src.evaluation.visualizer import AudioVisualizer

# 创建可视化器
visualizer = AudioVisualizer(save_dir="my_results")

# 生成对比图
visualizer.plot_metrics_comparison(results_dict)
visualizer.plot_radar_chart(results_dict)
visualizer.generate_comprehensive_report(results_dict)
```

### 方法对比分析

```python
from src.evaluation.comparator import MethodComparator

# 创建对比器
comparator = MethodComparator()

# 进行对比分析
comparison_results = comparator.compare_methods(results_dict)

# 生成对比表格
table = comparator.generate_comparison_table()
print(table)

# 保存详细报告
comparator.save_comparison_report("comparison_report.json")
```

## 支持的方法

### 传统方法

1. **移频移向法 (frequency_shift)**
   - 原理: 通过频率偏移破坏反馈条件
   - 优点: 计算简单，实时性好
   - 缺点: 可能引入音调变化

2. **增益抑制法 (gain_suppression)**
   - 原理: 检测并抑制啸叫频段
   - 优点: 针对性强，效果明显
   - 缺点: 可能影响有用信号

3. **自适应反馈抵消法 (adaptive_feedback)**
   - 原理: 自适应滤波器估计并消除反馈
   - 优点: 自适应强，效果稳定
   - 缺点: 计算复杂度较高

### 深度学习方法

1. **UNet方法 (unet)**
   - 原理: 端到端深度学习降噪
   - 优点: 效果好，适应性强
   - 缺点: 需要训练数据，计算量大

## 输出结果

### 评估报告结构

```json
{
  "test_summary": {
    "total_methods": 4,
    "methods_tested": ["frequency_shift", "gain_suppression", "adaptive_feedback", "unet"],
    "test_timestamp": "2024-01-01 12:00:00",
    "device": "cuda:0",
    "batch_size": 4
  },
  "performance_summary": {
    "snr_improvement_db": {
      "best_method": "unet",
      "best_value": 12.5,
      "average": 9.3
    }
  },
  "recommendations": {
    "best_overall": "unet",
    "best_quality": "unet",
    "most_efficient": "gain_suppression",
    "best_for_realtime": "gain_suppression"
  }
}
```

### 可视化图表

1. **指标对比图**: 各方法在不同指标上的表现
2. **雷达图**: 多维度性能对比
3. **计算效率图**: 时间、内存、参数量对比
4. **综合报告**: 包含所有图表的完整报告

## 使用示例

### 示例1: 快速测试传统方法

```python
from src.evaluation.test_runner import run_quick_evaluation

# 快速测试所有传统方法
results = run_quick_evaluation(
    methods=['frequency_shift', 'gain_suppression', 'adaptive_feedback'],
    num_samples=10
)

print("快速测试结果:")
for method, metrics in results['quick_results'].items():
    print(f"{method}: SNR改善={metrics['snr_improvement_db']:.2f}dB")
```

### 示例2: 对比自定义方法

```python
from src.evaluation.comparator import MethodComparator

# 自定义方法结果
my_results = {
    'my_method': {
        'snr_improvement_db': 10.5,
        'psnr_db': 28.5,
        'stoi_score': 0.87,
        'processing_time_ms': 45.6
    },
    'baseline': {
        'snr_improvement_db': 5.2,
        'psnr_db': 20.1,
        'stoi_score': 0.72,
        'processing_time_ms': 12.3
    }
}

# 进行对比
comparator = MethodComparator()
comparison = comparator.compare_methods(my_results)

print(f"最佳方法: {comparison['recommendations']['best_overall']}")
```

### 示例3: 生成可视化报告

```python
from src.evaluation.visualizer import AudioVisualizer

# 模拟结果数据
results = {
    'method_a': {'snr_improvement_db': 8.5, 'psnr_db': 25.3, 'stoi_score': 0.82},
    'method_b': {'snr_improvement_db': 9.1, 'psnr_db': 26.8, 'stoi_score': 0.85}
}

# 生成可视化
visualizer = AudioVisualizer()
visualizer.plot_metrics_comparison(results)
visualizer.plot_radar_chart(results)
visualizer.generate_comprehensive_report(results)
```

## 配置参数

### 评估配置

```python
# 在src/config.py中修改
SAMPLE_RATE = 16000          # 采样率
CHUNK_LEN = 3               # 音频片段长度(秒)
N_FFT = 512                 # FFT窗口大小
HOP_LENGTH = 128             # 跳跃长度
BATCH_SIZE = 8               # 批大小
NUM_WORKERS = 2             # 数据加载线程数
```

### 方法参数

```python
# 移频移向法参数
frequency_shift_params = {
    'shift_hz': 20.0,        # 频率偏移量(Hz)
    'sample_rate': 16000     # 采样率
}

# 增益抑制法参数
gain_suppression_params = {
    'threshold_db': -30.0,   # 抑制阈值(dB)
    'reduction_factor': 0.1   # 抑制因子
}

# 自适应反馈抵消法参数
adaptive_feedback_params = {
    'filter_length': 64,     # 滤波器长度
    'step_size': 0.01,       # 步长
    'leakage': 0.99         # 泄漏因子
}
```

## 故障排除

### 常见问题

1. **内存不足**
   ```
   解决方案: 减少batch_size或使用CPU模式
   ```

2. **CUDA错误**
   ```
   解决方案: 检查CUDA安装或使用CPU模式
   ```

3. **数据加载失败**
   ```
   解决方案: 检查数据路径和文件格式
   ```

4. **可视化失败**
   ```
   解决方案: 安装matplotlib和seaborn
   pip install matplotlib seaborn
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行小规模测试
results = run_quick_evaluation(num_samples=1)
```

## 扩展开发

### 添加新方法

1. 在`src/traditional/`目录下创建新方法文件
2. 实现标准接口:
   ```python
   class NewMethod:
       def __call__(self, noisy_mag):
           # 处理逻辑
           return enhanced_mag
   ```
3. 在`benchmark.py`中注册新方法

### 添加新指标

1. 在`src/evaluation/metrics.py`中添加新指标函数
2. 更新`calculate_all_metrics`方法
3. 在可视化器中添加对应的图表

## 依赖包

```
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
pandas>=1.3.0
tqdm>=4.62.0
psutil>=5.8.0
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进本评估系统。

## 联系方式

如有问题或建议，请通过以下方式联系:
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本评估系统主要用于研究和教育目的，在实际应用中请根据具体需求进行调整和验证。
