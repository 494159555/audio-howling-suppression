#!/usr/bin/env python3
"""
统一测试脚本 - 音频啸叫抑制项目测试套件

支持多种测试模式：
- quick: 快速环境检查（导入、数据可用性）
- evaluation: 传统方法性能评估
- full: 全面测试（包括深度学习模型）

使用方法:
    python tests/run_tests.py --mode quick
    python tests/run_tests.py --mode evaluation
    python tests/run_tests.py --mode full
    python tests/run_tests.py --modules traditional,evaluation
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加src到Python路径
project_root = Path(__file__).parent.parent
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
        import traceback
        traceback.print_exc()
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
        
        all_exist = True
        for path in data_paths:
            if path.exists():
                file_count = len(list(path.glob("*.wav")))
                print(f"[OK] {path}: {file_count} 个文件")
            else:
                print(f"[WARN] {path}: 目录不存在")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"[ERROR] 数据检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traditional_methods():
    """测试传统方法基本功能"""
    print("\n测试传统方法基本功能...")
    
    try:
        from src.traditional import FrequencyShiftMethod, GainSuppressionMethod, AdaptiveFeedbackMethod
        
        # 创建测试数据
        batch_size, freq_bins, time_frames = 2, 256, 100
        test_data = torch.randn(batch_size, 1, freq_bins, time_frames).abs()
        test_data = torch.log10(test_data + 1e-8)
        
        print(f"测试数据: {test_data.shape}, 范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
        
        # 测试移频移向法
        freq_method = FrequencyShiftMethod(shift_hz=20.0)
        result1 = freq_method(test_data)
        print(f"[OK] 移频移向法: {result1.shape}, 范围: [{result1.min():.3f}, {result1.max():.3f}]")
        
        # 测试增益抑制法
        gain_method = GainSuppressionMethod(threshold_db=-30.0)
        result2 = gain_method(test_data)
        print(f"[OK] 增益抑制法: {result2.shape}, 范围: [{result2.min():.3f}, {result2.max():.3f}]")
        
        # 测试自适应反馈抵消法
        adaptive_method = AdaptiveFeedbackMethod(filter_length=64)
        result3 = adaptive_method(test_data)
        print(f"[OK] 自适应反馈抵消法: {result3.shape}, 范围: [{result3.min():.3f}, {result3.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 传统方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traditional_methods_evaluation():
    """测试传统方法的评估性能"""
    print("\n" + "="*60)
    print("测试传统方法性能评估")
    print("="*60)
    
    try:
        from src.evaluation.metrics import AudioMetrics
        from src.traditional import FrequencyShiftMethod, GainSuppressionMethod, AdaptiveFeedbackMethod
        
        # 创建评估器
        metrics = AudioMetrics()
        
        # 创建传统方法
        methods = {
            'frequency_shift': FrequencyShiftMethod(),
            'gain_suppression': GainSuppressionMethod(),
            'adaptive_feedback': AdaptiveFeedbackMethod()
        }
        
        # 创建测试数据
        batch_size = 2
        freq_bins = 256
        time_frames = 100
        
        # 模拟log域频谱图数据
        clean_spectrogram = torch.randn(batch_size, 1, freq_bins, time_frames).abs()
        clean_spectrogram = torch.log10(clean_spectrogram + 1e-8)
        
        # 添加啸叫（高频增强）
        noisy_spectrogram = clean_spectrogram.clone()
        noisy_spectrogram[:, :, 200:, :] += 0.5
        
        print(f"测试数据形状: {clean_spectrogram.shape}")
        print(f"纯净信号范围: [{clean_spectrogram.min():.3f}, {clean_spectrogram.max():.3f}]")
        print(f"带噪信号范围: [{noisy_spectrogram.min():.3f}, {noisy_spectrogram.max():.3f}]")
        
        # 测试每个方法
        results = {}
        
        for method_name, method in methods.items():
            print(f"\n测试 {method_name}...")
            
            try:
                # 处理信号
                enhanced = method(noisy_spectrogram)
                
                # 计算指标
                method_metrics = {}
                
                # SNR改善
                snr_improvement = metrics.calculate_snr(clean_spectrogram, enhanced, noisy_spectrogram)
                method_metrics['snr_improvement_db'] = snr_improvement
                
                # PSNR
                psnr = metrics.calculate_psnr(clean_spectrogram, enhanced)
                method_metrics['psnr_db'] = psnr
                
                # STOI（简化版本）
                stoi = metrics.calculate_stoi(clean_spectrogram, enhanced)
                method_metrics['stoi_score'] = stoi
                
                # 啸叫抑制指标
                howling_metrics = metrics.calculate_howling_reduction(noisy_spectrogram, enhanced)
                method_metrics.update(howling_metrics)
                
                # 计算效率指标
                comp_metrics = metrics.calculate_computational_metrics(
                    method_name, noisy_spectrogram, method
                )
                method_metrics.update(comp_metrics)
                
                results[method_name] = method_metrics
                
                print(f"  SNR改善: {snr_improvement:.2f} dB")
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  STOI: {stoi:.3f}")
                print(f"  啸叫衰减: {howling_metrics['howling_reduction_db']:.2f} dB")
                print(f"  处理时间: {comp_metrics['processing_time_ms']:.2f} ms")
                print(f"  内存使用: {comp_metrics['memory_usage_mb']:.2f} MB")
                
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 输出对比结果
        print("\n" + "="*60)
        print("方法对比结果")
        print("="*60)
        
        metrics_to_compare = ['snr_improvement_db', 'psnr_db', 'stoi_score', 
                             'howling_reduction_db', 'processing_time_ms']
        
        for metric in metrics_to_compare:
            print(f"\n{metric}:")
            for method_name, method_results in results.items():
                if metric in method_results:
                    value = method_results[metric]
                    print(f"  {method_name}: {value:.3f}")
        
        # 找出最佳方法
        print("\n" + "="*60)
        print("最佳方法推荐")
        print("="*60)
        
        best_methods = {}
        
        # SNR改善最佳
        best_snr = max(results.items(), key=lambda x: x[1].get('snr_improvement_db', -float('inf')))
        best_methods['SNR改善'] = best_snr[0]
        
        # 啸叫抑制最佳
        best_howling = max(results.items(), key=lambda x: x[1].get('howling_reduction_db', -float('inf')))
        best_methods['啸叫抑制'] = best_howling[0]
        
        # 速度最佳
        fastest = min(results.items(), key=lambda x: x[1].get('processing_time_ms', float('inf')))
        best_methods['处理速度'] = fastest[0]
        
        for category, method in best_methods.items():
            print(f"  {category}: {method}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_learning_models():
    """测试深度学习模型"""
    print("\n测试深度学习模型...")
    
    try:
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
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_system():
    """测试评估系统"""
    print("\n测试评估系统...")
    
    try:
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
        import traceback
        traceback.print_exc()
        return False


def run_predefined_tests(mode):
    """运行预定义的测试模式"""
    print("="*80)
    print(f"音频啸叫抑制项目测试 - 模式: {mode.upper()}")
    print("="*80)
    
    test_suites = {
        'quick': [
            ("模块导入", test_imports),
            ("数据可用性", test_data_availability)
        ],
        'evaluation': [
            ("传统方法评估", test_traditional_methods_evaluation)
        ],
        'full': [
            ("模块导入", test_imports),
            ("数据可用性", test_data_availability),
            ("传统方法", test_traditional_methods),
            ("深度学习模型", test_deep_learning_models),
            ("评估系统", test_evaluation_system)
        ]
    }
    
    tests = test_suites.get(mode, [])
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name}测试出现异常: {e}")
            import traceback
            traceback.print_exc()
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
        print("\n[SUCCESS] 所有测试通过！")
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")
    
    return passed == len(results)


def run_custom_tests(modules):
    """运行自定义测试模块"""
    print("="*80)
    print(f"音频啸叫抑制项目测试 - 自定义模块: {', '.join(modules)}")
    print("="*80)
    
    available_tests = {
        'imports': ("模块导入", test_imports),
        'data': ("数据可用性", test_data_availability),
        'traditional': ("传统方法", test_traditional_methods),
        'traditional_eval': ("传统方法评估", test_traditional_methods_evaluation),
        'models': ("深度学习模型", test_deep_learning_models),
        'evaluation': ("评估系统", test_evaluation_system)
    }
    
    results = []
    for module in modules:
        if module in available_tests:
            test_name, test_func = available_tests[module]
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"[ERROR] {test_name}测试出现异常: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        else:
            print(f"[WARN] 未知模块: {module}")
    
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
        print("\n[SUCCESS] 所有测试通过！")
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")
    
    return passed == len(results)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频啸叫抑制项目测试脚本')
    parser.add_argument('--mode', choices=['quick', 'evaluation', 'full'], default='quick',
                        help='测试模式: quick(快速检查), evaluation(性能评估), full(全面测试)')
    parser.add_argument('--modules', nargs='+', 
                        help='自定义测试模块: imports, data, traditional, traditional_eval, models, evaluation')
    
    args = parser.parse_args()
    
    if args.modules:
        success = run_custom_tests(args.modules)
    else:
        success = run_predefined_tests(args.mode)
    
    # 提供使用建议
    if success:
        print("\n使用建议:")
        print("1. 快速评估传统方法: python -m tests.run_tests --mode evaluation")
        print("2. 全面测试所有功能: python -m tests.run_tests --mode full")
        print("3. 自定义测试模块: python -m tests.run_tests --modules imports,traditional")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())