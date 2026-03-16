#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境和基础功能
"""

import sys
from pathlib import Path

# 添加src到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试基础配置
        from src.config import cfg
        print("[OK] 配置模块导入成功")
        print(f"   设备: {cfg.DEVICE}")
        print(f"   采样率: {cfg.SAMPLE_RATE}")
        print(f"   批大小: {cfg.BATCH_SIZE}")
        
        # 测试数据集
        from src.dataset import HowlingDataset
        print("[OK] 数据集模块导入成功")
        
        # 测试传统方法
        from src.traditional import FrequencyShiftMethod, GainSuppressionMethod, AdaptiveFeedbackMethod
        print("[OK] 传统方法模块导入成功")
        
        # 测试深度学习模型
        from src.models import AudioUNet5, AudioUNet3, AudioCNN, AudioRNN
        print("[OK] 深度学习模型模块导入成功")
        
        # 测试评估系统
        from src.evaluation import AudioMetrics, AudioVisualizer, MethodComparator
        print("[OK] 评估系统模块导入成功")
        
        # 测试测试运行器
        from src.evaluation.test_runner import run_quick_evaluation
        print("[OK] 测试运行器模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 模块导入失败: {e}")
        return False


def test_data_availability():
    """测试数据可用性"""
    print("\n测试数据可用性...")
    
    try:
        from src.config import cfg
        
        # 检查数据目录
        data_paths = [
            cfg.TRAIN_CLEAN_DIR,
            cfg.TRAIN_NOISY_DIR,
            cfg.VAL_CLEAN_DIR,
            cfg.VAL_NOISY_DIR
        ]
        
        for path in data_paths:
            if path.exists():
                file_count = len(list(path.glob("*.wav")))
                print(f"[OK] {path}: {file_count} 个文件")
            else:
                print(f"[ERROR] {path}: 目录不存在")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 数据检查失败: {e}")
        return False


def test_traditional_methods():
    """测试传统方法"""
    print("\n测试传统方法...")
    
    try:
        import torch
        from src.traditional import FrequencyShiftMethod, GainSuppressionMethod, AdaptiveFeedbackMethod
        from src.config import cfg
        
        # 创建测试数据
        batch_size, freq_bins, time_frames = 2, 256, 128
        test_data = torch.randn(batch_size, 1, freq_bins, time_frames)
        
        # 测试移频移向法
        freq_method = FrequencyShiftMethod(shift_hz=20.0)
        result1 = freq_method(test_data)
        print(f"[OK] 移频移向法: 输入{test_data.shape} -> 输出{result1.shape}")
        
        # 测试增益抑制法
        gain_method = GainSuppressionMethod(threshold_db=-30.0)
        result2 = gain_method(test_data)
        print(f"[OK] 增益抑制法: 输入{test_data.shape} -> 输出{result2.shape}")
        
        # 测试自适应反馈抵消法
        adaptive_method = AdaptiveFeedbackMethod(filter_length=64)
        result3 = adaptive_method(test_data)
        print(f"[OK] 自适应反馈抵消法: 输入{test_data.shape} -> 输出{result3.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 传统方法测试失败: {e}")
        return False


def test_deep_learning_models():
    """测试深度学习模型"""
    print("\n测试深度学习模型...")
    
    try:
        import torch
        from src.models import AudioUNet5, AudioUNet3, AudioCNN, AudioRNN
        from src.config import cfg
        
        device = cfg.DEVICE
        batch_size, freq_bins, time_frames = 2, 256, 128
        test_data = torch.randn(batch_size, 1, freq_bins, time_frames).to(device)
        
        models = [
            ("AudioUNet3", AudioUNet3()),
            ("AudioUNet5", AudioUNet5()),
            ("AudioCNN", AudioCNN()),
            ("AudioRNN", AudioRNN())
        ]
        
        for name, model in models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                result = model(test_data)
            print(f"[OK] {name}: 输入{test_data.shape} -> 输出{result.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 深度学习模型测试失败: {e}")
        return False


def test_evaluation_system():
    """测试评估系统"""
    print("\n测试评估系统...")
    
    try:
        import torch
        from src.evaluation import AudioMetrics, AudioVisualizer, MethodComparator
        
        # 创建测试数据
        batch_size, freq_bins, time_frames = 2, 256, 128
        clean = torch.randn(batch_size, 1, freq_bins, time_frames)
        noisy = clean + 0.1 * torch.randn_like(clean)
        enhanced = clean + 0.05 * torch.randn_like(clean)
        
        # 测试指标计算
        metrics_calculator = AudioMetrics()
        snr = metrics_calculator.calculate_snr(clean, enhanced, noisy)
        psnr = metrics_calculator.calculate_psnr(clean, enhanced)
        stoi = metrics_calculator.calculate_stoi(clean, enhanced)
        
        print(f"[OK] 指标计算: SNR={snr:.2f}dB, PSNR={psnr:.2f}dB, STOI={stoi:.3f}")
        
        # 测试可视化器
        visualizer = AudioVisualizer(save_dir="test_visualizations")
        print("[OK] 可视化器创建成功")
        
        # 测试对比器
        comparator = MethodComparator()
        print("[OK] 对比器创建成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 评估系统测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*80)
    print("音频啸叫抑制项目环境测试")
    print("="*80)
    
    tests = [
        ("模块导入", test_imports),
        ("数据可用性", test_data_availability),
        ("传统方法", test_traditional_methods),
        ("深度学习模型", test_deep_learning_models),
        ("评估系统", test_evaluation_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "="*80)
    print("测试结果摘要")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("\n[SUCCESS] 所有测试通过！环境配置正确，可以开始实验。")
        print("\n使用方法:")
        print("1. 快速评估: python run_experiment.py --mode quick")
        print("2. 全面评估: python run_experiment.py --mode comprehensive")
        print("3. 仅传统方法: python run_experiment.py --mode traditional")
        print("4. 环境测试: python tests/integration/quick_test.py")
    else:
        print("\n[WARNING] 部分测试失败，请检查环境配置。")


if __name__ == "__main__":
    main()
