'''
传统方法测试脚本

测试三种传统音频啸叫抑制方法的基本功能
'''

import torch
import sys
import os
from pathlib import Path

# 添加src到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from src.traditional.frequency_shift import FrequencyShiftMethod
from src.traditional.gain_suppression import GainSuppressionMethod
from src.traditional.adaptive_feedback import AdaptiveFeedbackMethod


def test_methods():
    """测试三种方法的基本功能"""
    print("测试传统音频啸叫抑制方法...")
    print("=" * 50)
    
    # 创建测试数据 [batch_size, channels, freq_bins, time_frames]
    test_input = torch.randn(2, 1, 256, 100).abs()
    test_input = torch.log10(test_input + 1e-8)  # log域
    
    print(f"测试数据: {test_input.shape}, 范围: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # 测试移频移向法
    try:
        method = FrequencyShiftMethod(shift_hz=20.0)
        output = method(test_input)
        print(f"[OK] 移频移向法: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"[ERROR] 移频移向法错误: {e}")
    
    # 测试增益抑制法
    try:
        method = GainSuppressionMethod(threshold_db=-30.0)
        output = method(test_input)
        print(f"[OK] 增益抑制法: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"[ERROR] 增益抑制法错误: {e}")
    
    # 测试自适应反馈抵消法
    try:
        method = AdaptiveFeedbackMethod(filter_length=64)
        output = method(test_input)
        print(f"[OK] 自适应反馈抵消法: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"[ERROR] 自适应反馈抵消法错误: {e}")
    
    print("=" * 50)
    print("测试完成！")


if __name__ == "__main__":
    test_methods()
