#!/usr/bin/env python3
"""模型测试脚本

本脚本用于验证项目中的所有U-Net模型变体是否可以正确导入、初始化和运行。

主要功能：
    - 测试所有已注册的U-Net模型
    - 验证模型前向传播是否正常
    - 统计每个模型的参数数量
    - 输出详细的测试报告

使用方法：
    python scripts/test_models.py

适用场景：
    - 添加新模型后验证是否正确集成
    - 环境迁移后检查模型是否可用
    - 快速查看各模型的参数量对比

作者：音频处理实验室
版本：1.0
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径，确保可以导入src模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models import (
    AudioUNet3,
    AudioUNet5,
    AudioUNet5Attention,
    AudioUNet5Residual,
    AudioUNet5Dilated,
    AudioUNet5Optimized,
)


def format_number(num: int) -> str:
    """格式化数字，添加千分位分隔符

    Args:
        num: 要格式化的整数

    Returns:
        格式化后的字符串，如 "1,234,567"
    """
    return f"{num:,}"


def test_single_model(model_class, model_name: str, sample_input: torch.Tensor) -> dict:
    """测试单个模型

    执行以下测试：
    1. 模型初始化
    2. 前向传播
    3. 参数统计

    Args:
        model_class: 模型类（如AudioUNet5）
        model_name: 模型显示名称
        sample_input: 测试用的输入张量

    Returns:
        包含测试结果的字典：
        {
            'name': 模型名称,
            'success': 是否成功,
            'params': 参数数量,
            'output_shape': 输出形状,
            'error': 错误信息（如果失败）
        }
    """
    result = {
        'name': model_name,
        'success': False,
        'params': 0,
        'output_shape': None,
        'error': None
    }

    try:
        # 步骤1: 初始化模型
        model = model_class()

        # 步骤2: 统计参数量
        params = sum(p.numel() for p in model.parameters())
        result['params'] = params

        # 步骤3: 前向传播测试
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度，节省内存
            output = model(sample_input)

        # 记录输出形状
        result['output_shape'] = tuple(output.shape)
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def print_test_results(results: list) -> None:
    """打印测试结果表格

    Args:
        results: 测试结果列表
    """
    print("\n" + "="*100)
    print("📊 模型测试结果汇总")
    print("="*100 + "\n")

    # 打印表头
    print(f"{'模型名称':<30} {'状态':<10} {'参数量':<15} {'输入形状':<20} {'输出形状':<20}")
    print("-"*100)

    # 打印每个模型的结果
    for result in results:
        if result['success']:
            status = "✅ 通过"
            params_str = format_number(result['params'])
            input_shape = str((2, 1, 256, 100))  # 样本输入形状
            output_shape = str(result['output_shape'])
        else:
            status = "❌ 失败"
            params_str = "N/A"
            input_shape = "N/A"
            output_shape = result['error'][:30] if result['error'] else "Unknown error"

        print(f"{result['name']:<30} {status:<10} {params_str:<15} {input_shape:<20} {output_shape:<20}")

    # 统计信息
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    print("-"*100)
    print(f"测试通过率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count > 0:
        params_list = [r['params'] for r in results if r['success']]
        min_params = min(params_list)
        max_params = max(params_list)
        avg_params = sum(params_list) // len(params_list)

        print(f"参数量统计: 最少={format_number(min_params)}, "
              f"最多={format_number(max_params)}, "
              f"平均={format_number(avg_params)}")

    print("="*100 + "\n")


def main():
    """主函数：执行所有模型的测试"""
    print("\n" + "="*100)
    print("🧪 U-Net模型测试脚本")
    print("="*100)
    print(f"\n📝 测试说明:")
    print(f"   - 测试所有已注册的U-Net模型变体")
    print(f"   - 验证模型初始化和前向传播")
    print(f"   - 统计每个模型的参数量")
    print(f"   - 输入形状: [Batch=2, Channels=1, Freq=256, Time=100]")

    # 定义要测试的模型列表
    # 格式: (模型类, 显示名称)
    models_to_test = [
        (AudioUNet3, "AudioUNet3 (3层基线U-Net)"),
        (AudioUNet5, "AudioUNet5 (5层基线U-Net)"),
        (AudioUNet5Attention, "AudioUNet5Attention (+注意力门)"),
        (AudioUNet5Residual, "AudioUNet5Residual (+残差连接)"),
        (AudioUNet5Dilated, "AudioUNet5Dilated (+空洞卷积)"),
        (AudioUNet5Optimized, "AudioUNet5Optimized (综合改进版)"),
    ]

    print(f"\n📋 待测试模型数量: {len(models_to_test)}")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    print(f"💻 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")

    # 创建测试输入
    # 形状: [Batch=2, Channels=1, Frequency=256, Time=100]
    # 这是频谱图的标准输入格式
    sample_input = torch.randn(2, 1, 256, 100)
    print(f"\n📊 样本输入形状: {tuple(sample_input.shape)}")
    print(f"   解释: 批次=2, 通道=1, 频率bin=256, 时间帧=100")

    # 执行测试
    print(f"\n⏳ 开始测试...")
    results = []

    for i, (model_class, model_name) in enumerate(models_to_test, 1):
        print(f"\n[{i}/{len(models_to_test)}] 测试: {model_name}")

        result = test_single_model(model_class, model_name, sample_input)
        results.append(result)

        if result['success']:
            print(f"   ✅ 成功 | 参数量: {format_number(result['params'])} | "
                  f"输出形状: {result['output_shape']}")
        else:
            print(f"   ❌ 失败 | 错误: {result['error']}")

    # 打印汇总表
    print_test_results(results)

    # 打印结论
    print("="*100)
    print("📌 测试结论")
    print("="*100)

    success_count = sum(1 for r in results if r['success'])

    if success_count == len(results):
        print("\n✅ 恭喜！所有模型测试通过！")
        print("\n💡 您可以使用以下任意模型进行训练:")
        print("   python src/train.py --model unet_v2          # 使用5层基线U-Net")
        print("   python src/train.py --model unet_v6_optimized # 使用综合改进版")
    else:
        failed_count = len(results) - success_count
        print(f"\n⚠️  警告: {failed_count} 个模型测试失败")
        print("\n💡 请检查:")
        print("   1. 模型代码是否有语法错误")
        print("   2. 模型是否正确导入到 src/models/__init__.py")
        print("   3. 模型是否在 src/config.py 中正确注册")

    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()