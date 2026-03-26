#!/usr/bin/env python3
"""音频啸叫抑制项目测试套件

本脚本提供全面的自动化测试，验证项目各模块的正确性。

测试模式：
    - quick: 快速检查（导入、数据）
    - evaluation: 传统方法评估
    - full: 全面测试
    - models: 模型和损失函数测试

使用方法：
    # 快速测试
    python tests/run_tests.py

    # 指定测试模式
    python tests/run_tests.py --mode full

    # 自定义测试模块
    python tests/run_tests.py --modules imports unet_models

作者：音频处理实验室
版本：3.0.0
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加src到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))


# ==================== 测试函数 ====================

def test_imports():
    """测试1: 模块导入测试

    验证所有核心模块能否正确导入
    """
    print("\n" + "="*60)
    print("📦 测试1: 模块导入")
    print("="*60)

    try:
        # 基础配置
        from src.config import cfg
        print("✅ 配置模块导入成功")
        print(f"   设备: {cfg.DEVICE}")
        print(f"   采样率: {cfg.SAMPLE_RATE} Hz")

        # 数据集
        from src.dataset import HowlingDataset
        print("✅ 数据集模块导入成功")

        # 传统方法
        from src.traditional import (
            FrequencyShiftMethod,
            GainSuppressionMethod,
            AdaptiveFeedbackMethod
        )
        print("✅ 传统方法模块导入成功")

        # 深度学习模型
        from src.models import AudioUNet5, AudioUNet3, AudioCNN, AudioRNN
        print("✅ 深度学习模型模块导入成功")

        # 评估系统
        from src.evaluation import AudioMetrics, AudioVisualizer, MethodComparator
        print("✅ 评估系统模块导入成功")

        return True

    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_data_availability():
    """测试2: 数据可用性检查

    验证训练/验证/测试数据目录是否存在及文件数量
    """
    print("\n" + "="*60)
    print("📊 测试2: 数据可用性")
    print("="*60)

    try:
        from src.config import cfg

        data_paths = {
            "训练集(纯净)": cfg.TRAIN_CLEAN_DIR,
            "训练集(啸叫)": cfg.TRAIN_NOISY_DIR,
            "验证集(纯净)": cfg.VAL_CLEAN_DIR,
            "验证集(啸叫)": cfg.VAL_NOISY_DIR
        }

        all_exist = True
        for name, path in data_paths.items():
            if path.exists():
                file_count = len(list(path.glob("*.wav")))
                print(f"✅ {name}: {file_count} 个文件")
            else:
                print(f"⚠️  {name}: 目录不存在 ({path})")
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        return False


def test_traditional_methods():
    """测试3: 传统方法基本功能

    测试频率移位、增益抑制、自适应反馈方法的基本功能
    """
    print("\n" + "="*60)
    print("🔧 测试3: 传统方法基本功能")
    print("="*60)

    try:
        from src.traditional import (
            FrequencyShiftMethod,
            GainSuppressionMethod,
            AdaptiveFeedbackMethod
        )

        # 创建测试数据
        test_data = torch.randn(2, 1, 256, 100).abs()
        test_data = torch.log10(test_data + 1e-8)
        print(f"测试数据: {test_data.shape}")

        # 测试频率移位
        freq_method = FrequencyShiftMethod(shift_hz=20.0)
        result1 = freq_method(test_data)
        print(f"✅ 移频移向法: {result1.shape}")

        # 测试增益抑制
        gain_method = GainSuppressionMethod(threshold_db=-30.0)
        result2 = gain_method(test_data)
        print(f"✅ 增益抑制法: {result2.shape}")

        # 测试自适应反馈
        adaptive_method = AdaptiveFeedbackMethod(filter_length=64)
        result3 = adaptive_method(test_data)
        print(f"✅ 自适应反馈法: {result3.shape}")

        return True

    except Exception as e:
        print(f"❌ 传统方法测试失败: {e}")
        return False


def test_deep_learning_models():
    """测试4: 深度学习模型

    测试主要深度学习模型的初始化和前向传播
    """
    print("\n" + "="*60)
    print("🧠 测试4: 深度学习模型")
    print("="*60)

    try:
        from src.models import AudioUNet5, AudioUNet3, AudioCNN, AudioRNN
        from src.config import cfg

        device = cfg.DEVICE
        test_data = torch.randn(2, 1, 256, 128).to(device)

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
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {name:<15} 参数: {params:>10,}  输出: {result.shape}")

        return True

    except Exception as e:
        print(f"❌ 深度学习模型测试失败: {e}")
        return False


def test_evaluation_system():
    """测试5: 评估系统

    测试指标计算、可视化、对比等评估功能
    """
    print("\n" + "="*60)
    print("📈 测试5: 评估系统")
    print("="*60)

    try:
        from src.evaluation import AudioMetrics, AudioVisualizer, MethodComparator

        # 创建测试数据
        clean = torch.randn(2, 1, 256, 128)
        noisy = clean + 0.1 * torch.randn_like(clean)
        enhanced = clean + 0.05 * torch.randn_like(clean)

        # 测试指标计算
        metrics = AudioMetrics()
        snr = metrics.calculate_snr(clean, enhanced, noisy)
        psnr = metrics.calculate_psnr(clean, enhanced)
        stoi = metrics.calculate_stoi(clean, enhanced)

        print(f"✅ 指标计算:")
        print(f"   SNR:  {snr:.2f} dB")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   STOI: {stoi:.3f}")

        # 测试可视化器
        visualizer = AudioVisualizer(save_dir="test_visualizations")
        print("✅ 可视化器初始化成功")

        # 测试对比器
        comparator = MethodComparator()
        print("✅ 对比器初始化成功")

        return True

    except Exception as e:
        print(f"❌ 评估系统测试失败: {e}")
        return False


def test_all_unet_models():
    """测试6: 所有U-Net模型变体

    测试所有U-Net模型变体的初始化和前向传播
    """
    print("\n" + "="*60)
    print("🔬 测试6: 所有U-Net模型变体")
    print("="*60)

    try:
        from src.models import (
            AudioUNet3,
            AudioUNet5,
            AudioUNet5Attention,
            AudioUNet5Residual,
            AudioUNet5Dilated,
            AudioUNet5Optimized,
            AudioUNet5LSTM,
            AudioUNet5TemporalAttention,
            AudioUNet5ConvLSTM,
            AudioUNet5GAN,
        )
        from src.config import cfg

        device = cfg.DEVICE
        models = [
            (AudioUNet3, "v1: 3层基线"),
            (AudioUNet5, "v2: 5层基线"),
            (AudioUNet5Attention, "v3: 注意力"),
            (AudioUNet5Residual, "v4: 残差"),
            (AudioUNet5Dilated, "v5: 空洞卷积"),
            (AudioUNet5Optimized, "v6: 综合优化"),
            (AudioUNet5LSTM, "v7: LSTM"),
            (AudioUNet5TemporalAttention, "v8: 时间注意力"),
            (AudioUNet5ConvLSTM, "v9: ConvLSTM"),
            (AudioUNet5GAN, "v10: GAN"),
        ]

        results = []
        for model_class, model_desc in models:
            try:
                model = model_class().to(device)
                x = torch.randn(1, 1, 256, 100).to(device)

                with torch.no_grad():
                    if hasattr(model, 'generator'):
                        output = model.generator(x)
                    else:
                        output = model(x)

                params = sum(p.numel() for p in model.parameters())
                print(f"✅ {model_desc:<20} 参数: {params:>10,}")
                results.append((model_desc, True))

            except Exception as e:
                print(f"❌ {model_desc:<20} 错误: {str(e)[:40]}")
                results.append((model_desc, False))

        passed = sum(1 for _, r in results if r)
        print(f"\n通过率: {passed}/{len(results)}")

        return all(r for _, r in results)

    except Exception as e:
        print(f"❌ U-Net模型测试失败: {e}")
        return False


def test_loss_functions():
    """测试7: 损失函数

    测试所有损失函数的初始化和计算
    """
    print("\n" + "="*60)
    print("📉 测试7: 损失函数")
    print("="*60)

    try:
        from src.models import (
            SpectralLoss,
            MultiTaskLoss,
            SpectralConsistencyLoss,
            AdversarialLoss,
            Discriminator,
        )

        loss_functions = [
            (SpectralLoss, "频谱损失"),
            (MultiTaskLoss, "多任务损失"),
            (SpectralConsistencyLoss, "频谱一致性损失"),
            (AdversarialLoss, "对抗损失"),
        ]

        results = []
        for loss_class, loss_name in loss_functions:
            try:
                loss_fn = loss_class()
                pred = torch.randn(2, 1, 256, 100).abs()
                target = torch.randn(2, 1, 256, 100).abs()

                loss = loss_fn(pred, target)
                print(f"✅ {loss_name:<20} Loss: {loss.item():.4f}")
                results.append((loss_name, True))

            except Exception as e:
                print(f"❌ {loss_name:<20} 错误: {str(e)[:40]}")
                results.append((loss_name, False))

        # 测试判别器
        try:
            disc = Discriminator()
            fake_pred = torch.randn(2, 1)
            real_pred = torch.randn(2, 1)
            d_loss = disc.discriminator_loss(real_pred, fake_pred)
            print(f"✅ {'判别器':<20} Loss: {d_loss.item():.4f}")
        except Exception as e:
            print(f"❌ {'判别器':<20} 错误: {str(e)[:40]}")

        passed = sum(1 for _, r in results if r)
        print(f"\n通过率: {passed}/{len(results)}")

        return all(r for _, r in results)

    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False


# ==================== 测试套件 ====================

TEST_SUITES = {
    'quick': [
        ("模块导入", test_imports),
        ("数据可用性", test_data_availability)
    ],
    'evaluation': [
        ("传统方法", test_traditional_methods),
        ("评估系统", test_evaluation_system)
    ],
    'full': [
        ("模块导入", test_imports),
        ("数据可用性", test_data_availability),
        ("传统方法", test_traditional_methods),
        ("深度学习模型", test_deep_learning_models),
        ("评估系统", test_evaluation_system)
    ],
    'models': [
        ("所有U-Net模型", test_all_unet_models),
        ("损失函数", test_loss_functions)
    ]
}


def run_tests(mode='quick'):
    """运行测试套件

    Args:
        mode: 测试模式 ('quick', 'evaluation', 'full', 'models')

    Returns:
        bool: 是否所有测试通过
    """
    print("\n" + "="*70)
    print("🧪 音频啸叫抑制项目 - 自动化测试")
    print("="*70)
    print(f"测试模式: {mode.upper()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")

    tests = TEST_SUITES.get(mode, [])
    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 打印测试摘要
    print("\n" + "="*70)
    print("📋 测试结果摘要")
    print("="*70)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！项目运行正常。")
    else:
        print(f"\n⚠️  {total - passed} 项测试失败，请检查上述错误信息。")

    print("="*70 + "\n")

    return passed == total


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='音频啸叫抑制项目自动化测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式说明:
  quick       快速检查（导入、数据）
  evaluation  传统方法和评估系统
  full        全面测试所有功能
  models      模型和损失函数测试

使用示例:
  python tests/run_tests.py                    # 默认快速测试
  python tests/run_tests.py --mode full        # 全面测试
  python tests/run_tests.py --mode models      # 测试所有模型
        """
    )

    parser.add_argument(
        '--mode',
        choices=['quick', 'evaluation', 'full', 'models'],
        default='quick',
        help='测试模式（默认: quick）'
    )

    args = parser.parse_args()

    # 运行测试
    success = run_tests(args.mode)

    # 提供使用建议
    if not success:
        print("\n💡 故障排查建议:")
        print("1. 检查数据目录是否存在: data/train, data/dev")
        print("2. 检查依赖是否安装: pip install -r requirements.txt")
        print("3. 查看详细错误信息以定位问题")
        print("4. 运行单模块测试: python tests/run_tests.py --mode models\n")
    else:
        print("\n💡 下一步:")
        print("1. 训练模型: python src/train.py --config configs/unet_v2.yaml")
        print("2. 查看脚本: python scripts/test_models.py")
        print("3. 运行实验: python scripts/run_experiment.py --mode quick\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
