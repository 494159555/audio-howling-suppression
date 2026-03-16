'''
评估模块 - 音频啸叫抑制方法科学评估框架

模块功能：
- 提供全面的音频质量评估指标
- 支持深度学习和传统方法的对比评估
- 生成可视化评估报告
- 提供基准测试和性能分析工具

主要组件：
- metrics: 各种评估指标实现
- visualizer: 可视化工具
- comparator: 方法对比工具
- benchmark: 基准测试脚本
- test_runner: 统一测试运行器

使用方法：
from src.evaluation import AudioEvaluator, run_comprehensive_evaluation

# 创建评估器
evaluator = AudioEvaluator()

# 运行全面评估
results = run_comprehensive_evaluation(
    methods=['unet', 'frequency_shift', 'gain_suppression', 'adaptive_feedback'],
    test_data_path='data/test'
)
'''

from .metrics import AudioMetrics
from .visualizer import AudioVisualizer
from .comparator import MethodComparator
from .benchmark import BenchmarkRunner
from .test_runner import run_comprehensive_evaluation

__all__ = [
    'AudioMetrics',
    'AudioVisualizer', 
    'MethodComparator',
    'BenchmarkRunner',
    'run_comprehensive_evaluation'
]
