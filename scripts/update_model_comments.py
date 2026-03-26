#!/usr/bin/env python3
"""模型文件注释更新工具

本脚本用于批量更新模型文件的文档字符串和注释，保持代码注释的一致性。

主要功能：
    - 为模型文件添加统一的文档字符串模板
    - 更新类和函数的注释说明
    - 确保所有模型文件遵循相同的注释规范
    - 自动生成模型描述信息

使用方法：
    # 更新所有模型文件
    python scripts/update_model_comments.py --all

    # 更新指定模型文件
    python scripts/update_model_comments.py --model unet_v2

    # 预览将要更新的内容（不实际修改）
    python scripts/update_model_comments.py --all --preview

    # 备份原始文件
    python scripts/update_model_comments.py --all --backup

输出：
    - 更新后的模型文件（带完整注释）
    - 可选：备份文件（.bak后缀）

注意事项：
    - 建议先使用 --preview 预览更改
    - 推荐使用 --backup 备份原始文件
    - 确保在Git版本控制下运行，方便回滚

作者：音频处理实验室
版本：1.0
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# 模型描述信息（用于生成文档字符串）
MODEL_DESCRIPTIONS = {
    'unet_v1': {
        'name': 'AudioUNet3',
        'title': '3层U-Net基线模型',
        'description': '''
3层U-Net架构，用于音频啸叫抑制。

这是最基础的U-Net实现，包含3层下采样和3层上采样。
适合用于快速原型开发和基线对比。

网络结构：
    - 编码器: 3层下采样，每层包含卷积、ReLU、最大池化
    - 瓶颈层: 卷积层提取深层特征
    - 解码器: 3层上采样，每层包含转置卷积、ReLU
    - 跳跃连接: 连接编码器和解码器的对应层

输入:
    - 形状: [batch, 1, 256, time]
    - 范围: [0, 1] 归一化的对数幅度谱

输出:
    - 形状: [batch, 1, 256, time]
    - 范围: [0, 1] 预测的乘性掩码

参数量: ~100K
'''
    },
    'unet_v2': {
        'name': 'AudioUNet5',
        'title': '5层U-Net基线模型',
        'description': '''
5层U-Net架构，用于音频啸叫抑制。

这是标准的U-Net实现，包含5层下采样和5层上采样。
相比3层版本有更强的特征提取能力，是本项目默认使用的模型。

网络结构：
    - 编码器: 5层下采样，每层包含卷积、ReLU、最大池化
    - 瓶颈层: 双卷积层提取深层特征
    - 解码器: 5层上采样，每层包含转置卷积、ReLU
    - 跳跃连接: 连接编码器和解码器的对应层

输入:
    - 形状: [batch, 1, 256, time]
    - 范围: [0, 1] 归一化的对数幅度谱

输出:
    - 形状: [batch, 1, 256, time]
    - 范围: [0, 1] 预测的乘性掩码

参数量: ~2M
'''
    },
    'unet_v3_attention': {
        'name': 'AudioUNet5Attention',
        'title': '5层U-Net + 注意力门',
        'description': '''
带注意力机制的5层U-Net架构。

在标准U-Net基础上添加了注意力门（Attention Gate）机制。
注意力门可以让模型学习关注输入中的重要区域，抑制不相关的信息。

改进点：
    - 注意力门: 在跳跃连接处添加注意力机制
    - 自适应加权: 根据特征重要性动态调整跳跃连接的贡献
    - 更好的特征融合: 注意力权重指导特征融合

网络结构：
    - 编码器: 5层下采样
    - 解码器: 5层上采样 + 注意力门
    - 注意力机制: 在每个跳跃连接处添加

优势：
    - 提高模型对关键特征的注意力
    - 减少无关信息的干扰
    - 通常能提升抑制效果

参数量: ~2.5M
'''
    },
    'unet_v4_residual': {
        'name': 'AudioUNet5Residual',
        'title': '5层U-Net + 残差连接',
        'description': '''
带残差连接的5层U-Net架构。

在标准U-Net基础上添加了残差连接（Residual Connection）。
残差连接可以缓解梯度消失问题，允许训练更深的网络。

改进点：
    - 残差块: 使用残差连接替代普通卷积块
    - 梯度传播: 更容易反向传播梯度
    - 网络深度: 支持更深的网络结构

残差块结构：
    input -> Conv1 -> ReLU -> Conv2 -> (+) -> ReLU -> output
              ^                           |
              |---------------------------|

优势：
    - 更容易训练深层网络
    - 减少梯度消失问题
    - 提升特征表达能力

参数量: ~2.2M
'''
    },
    'unet_v5_dilated': {
        'name': 'AudioUNet5Dilated',
        'title': '5层U-Net + 空洞卷积',
        'description': '''
带空洞卷积的5层U-Net架构。

使用空洞卷积（Dilated Convolution）替代部分标准卷积。
空洞卷积可以扩大感受野，捕捉更大范围的上下文信息。

改进点：
    - 空洞卷积: 使用不同的膨胀率
    - 大感受野: 不增加参数量的情况下扩大感受野
    - 多尺度特征: 不同的膨胀率捕捉不同尺度的特征

空洞卷积原理：
    - 膨胀率=1: 标准卷积
    - 膨胀率=2: 卷积核元素之间间隔1个位置
    - 膨胀率=4: 卷积核元素之间间隔3个位置

优势：
    - 扩大感受野
    - 保持参数量不变
    - 更好地捕捉长距离依赖

参数量: ~2M
'''
    },
    'unet_v6_optimized': {
        'name': 'AudioUNet5Optimized',
        'title': '5层U-Net 综合优化版',
        'description': '''
综合优化的5层U-Net架构。

结合了注意力机制、残差连接和空洞卷积的优点。
这是U-Net系列的优化版本，在性能和效率之间取得平衡。

改进点：
    - 注意力门: 关注重要特征区域
    - 残差连接: 改善梯度传播
    - 空洞卷积: 扩大感受野
    - 三合一设计: 综合上述所有改进

网络结构：
    - 编码器: 残差块 + 空洞卷积
    - 解码器: 残差块 + 注意力门
    - 跳跃连接: 带注意力的特征融合

优势：
    - 最佳的性能表现
    - 更好的特征提取能力
    - 适合复杂场景

参数量: ~2.8M
'''
    },
    'unet_v7_lstm': {
        'name': 'AudioUNet5LSTM',
        'title': '5层U-Net + 双向LSTM',
        'description': '''
结合时序建模的5层U-Net架构。

在U-Net的瓶颈层添加了双向LSTM（Long Short-Term Memory）。
LSTM可以捕捉音频的时序依赖关系，对于时变啸叫特别有效。

改进点：
    - 双向LSTM: 在瓶颈层添加LSTM层
    - 时序建模: 捕捉长时间依赖关系
    - 前向+后向: 同时利用过去和未来的信息

LSTM层位置：
    Encoder -> LSTM -> Decoder

优势：
    - 更好地建模时序信息
    - 适合处理时变啸叫
    - 提升时序连续性

参数量: ~3.5M
'''
    },
    'unet_v8_temporal_attention': {
        'name': 'AudioUNet5TemporalAttention',
        'title': '5层U-Net + 时间注意力',
        'description': '''
带时间注意力机制的5层U-Net架构。

添加了专门的时间注意力模块，让模型学习关注时间维度的重要片段。
时间注意力可以动态调整对不同时间帧的关注程度。

改进点：
    - 时间注意力: 在时间维度上应用注意力
    - 动态加权: 根据重要性调整时间帧权重
    - 自适应关注: 自动学习关键时间段

时间注意力机制：
    - 计算每个时间帧的重要性分数
    - 使用softmax归一化
    - 加权求和得到注意力输出

优势：
    - 关注重要的时间片段
    - 抑制不重要的噪声时段
    - 提升时序处理能力

参数量: ~2.3M
'''
    },
    'unet_v9_convlstm': {
        'name': 'AudioUNet5ConvLSTM',
        'title': '5层U-Net + ConvLSTM',
        'description': '''
结合ConvLSTM的5层U-Net架构。

ConvLSTM结合了CNN的空间特征提取能力和LSTM的时序建模能力。
特别适合处理音频频谱图这种既有空间（频率）又有时间维度数据。

改进点：
    - ConvLSTM: 在瓶颈层使用ConvLSTM
    - 时空建模: 同时捕捉空间和时间模式
    - 记忆机制: LSTM单元记住长期依赖

ConvLSTM vs LSTM:
    - LSTM: 处理1D序列
    - ConvLSTM: 处理2D+时间的数据

优势：
    - 同时建模空间和时间
    - 保留频谱结构信息
    - 更适合音频任务

参数量: ~4M
'''
    },
    'unet_v10_gan': {
        'name': 'AudioUNet5GAN',
        'title': '5层U-Net + GAN框架',
        'description': '''
基于生成对抗网络（GAN）的架构。

使用U-Net作为生成器，添加判别器网络形成对抗训练。
GAN可以生成更真实、更自然的输出音频。

架构组成：
    - 生成器（Generator）: U-Net模型
    - 判别器（Discriminator）: 判断音频是否真实
    - 对抗训练: 生成器和判别器相互竞争

训练目标：
    - 生成器: 生成能骗过判别器的音频
    - 判别器: 区分真实和生成的音频
    - 平衡点: 生成器生成逼真的音频

优势：
    - 生成质量更高
    - 音频更自然
    - 减少伪影

参数量:
    - 生成器: ~2M
    - 判别器: ~1.5M
'''
    },
    'unet_v11_multiscale': {
        'name': 'AudioUNet5MultiScale',
        'title': '多尺度U-Net',
        'description': '''
多尺度特征提取的U-Net架构。

同时使用3个不同深度的U-Net分支提取多尺度特征。
不同尺度的特征捕捉不同分辨率的音频模式。

多尺度设计：
    - 分支1: 3层U-Net（浅层，高频细节）
    - 分支2: 5层U-Net（中层，中频特征）
    - 分支3: 7层U-Net（深层，低频模式）
    - 融合: 加权融合所有分支的输出

优势：
    - 捕捉多尺度特征
    - 同时处理细节和全局模式
    - 更全面的特征表示

参数量: ~6M
'''
    },
    'unet_v12_pyramid': {
        'name': 'AudioUNet5Pyramid',
        'title': '金字塔池化U-Net',
        'description': '''
带金字塔池化模块（PPM）的U-Net架构。

在解码器中添加金字塔池化模块，聚合不同区域的上下文信息。
金字塔池化可以捕捉不同尺度的全局上下文。

金字塔池化模块：
    - 使用不同级别的池化（1x1, 2x2, 3x3, 6x6）
    - 每个级别提取不同尺度的全局信息
    - 上采样到原始大小后连接

优势：
    - 融合全局上下文信息
    - 提升对整体结构的理解
    - 改善局部和全局的平衡

参数量: ~2.5M
'''
    },
    'unet_v13_fpn': {
        'name': 'AudioUNet5FPN',
        'title': '特征金字塔网络U-Net',
        'description': '''
基于特征金字塔网络（FPN）的U-Net架构。

FPN设计了一个自顶向下的路径，将高层特征传递给低层。
使得每一层都能获得强语义信息和精细的空间细节。

FPN架构：
    - 自底向上路径: 编码器（提取特征）
    - 自顶向下路径: 上采样高层特征
    - 横向连接: 融合同尺度的特征
    - 每层输出: 融合后的多尺度特征

优势：
    - 多尺度特征融合
    - 强语义 + 精细节
    - 更好的特征表示

参数量: ~2.7M
'''
    },
}


def generate_model_docstring(model_key: str) -> str:
    """生成模型的文档字符串

    Args:
        model_key: 模型标识符（如 'unet_v2'）

    Returns:
        格式化的文档字符串
    """
    if model_key not in MODEL_DESCRIPTIONS:
        return f'''{model_key.upper()} 模型

音频啸叫抑制模型
'''

    info = MODEL_DESCRIPTIONS[model_key]

    docstring = f'''"""
{info['title']}

{info['description'].strip()}
"""'''

    return docstring


def update_file_header(file_path: Path, model_key: str, backup: bool = False) -> bool:
    """更新文件头部的文档字符串

    Args:
        file_path: 文件路径
        model_key: 模型标识符
        backup: 是否创建备份

    Returns:
        是否成功更新
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 创建备份
        if backup:
            backup_path = file_path.with_suffix('.py.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ 已创建备份: {backup_path}")

        # 生成新的文档字符串
        new_docstring = generate_model_docstring(model_key)

        # 查找并替换文件头部的文档字符串
        # 匹配模式：文件开头的 """..."""
        pattern = r'^""".*?"""'
        replacement = new_docstring

        new_content = re.sub(pattern, new_docstring, content, count=1, flags=re.DOTALL)

        # 如果没有找到现有的文档字符串，在文件开头添加
        if new_content == content:
            # 在第一行非注释行前插入
            lines = content.split('\n')
            insert_pos = 0

            for i, line in enumerate(lines):
                if not line.strip().startswith('#') and line.strip() != '':
                    insert_pos = i
                    break

            lines.insert(insert_pos, new_docstring)
            new_content = '\n'.join(lines)

        # 写入更新后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"  ✗ 更新失败: {e}")
        return False


def get_all_model_files() -> List[Path]:
    """获取所有模型文件

    Returns:
        模型文件路径列表
    """
    models_dir = Path(__file__).parent.parent / 'src' / 'models'

    if not models_dir.exists():
        print(f"错误: 模型目录不存在: {models_dir}")
        return []

    # 查找所有 unet_*.py, CNN.py, RNN.py
    model_files = []
    model_files.extend(models_dir.glob('unet_*.py'))
    model_files.extend(models_dir.glob('[CN][NN][A-Z]*.py'))  # CNN.py, RNN.py

    return sorted(model_files)


def extract_model_key(file_path: Path) -> str:
    """从文件名提取模型标识符

    Args:
        file_path: 文件路径

    Returns:
        模型标识符
    """
    filename = file_path.stem  # 不带扩展名的文件名

    # 处理特殊文件名
    if filename == 'CNN':
        return 'cnn'
    elif filename == 'RNN':
        return 'rnn'
    else:
        return filename


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='模型文件注释更新工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--all', action='store_true',
                       help='更新所有模型文件')

    parser.add_argument('--model', type=str,
                       help='更新指定的模型文件（如 unet_v2）')

    parser.add_argument('--preview', action='store_true',
                       help='预览模式，不实际修改文件')

    parser.add_argument('--backup', action='store_true',
                       help='创建备份文件（.py.bak）')

    args = parser.parse_args()

    # 确定要更新的文件列表
    model_files = []

    if args.all:
        model_files = get_all_model_files()
        if not model_files:
            print("未找到任何模型文件")
            return
    elif args.model:
        models_dir = Path(__file__).parent.parent / 'src' / 'models'
        possible_file = models_dir / f'{args.model}.py'
        if possible_file.exists():
            model_files = [possible_file]
        else:
            print(f"错误: 模型文件不存在: {possible_file}")
            return
    else:
        print("请使用 --all 或 --model 指定要更新的文件")
        print("使用 --help 查看帮助信息")
        return

    print(f"\n{'='*70}")
    print("📝 模型文件注释更新工具")
    print(f"{'='*70}\n")

    if args.preview:
        print("⚠️  预览模式：不会实际修改文件\n")

    if args.backup:
        print("💾 将为每个文件创建备份 (.py.bak)\n")

    # 更新每个文件
    success_count = 0
    fail_count = 0

    for file_path in model_files:
        model_key = extract_model_key(file_path)

        print(f"处理: {file_path.name}")

        if args.preview:
            # 只显示将要生成的文档字符串
            docstring = generate_model_docstring(model_key)
            print(f"  将添加/更新为:\n{docstring}\n")
            success_count += 1
        else:
            # 实际更新文件
            if update_file_header(file_path, model_key, args.backup):
                print(f"  ✓ 更新成功")
                success_count += 1
            else:
                print(f"  ✗ 更新失败")
                fail_count += 1
            print()

    # 打印总结
    print(f"{'='*70}")
    print(f"总结: 成功 {success_count}, 失败 {fail_count}, 总计 {len(model_files)}")
    print(f"{'='*70}\n")

    if not args.preview and success_count > 0:
        print("💡 提示:")
        print("   - 检查更新后的文件是否符合预期")
        print("   - 如果不满意，可以使用备份文件恢复")
        print("   - 或使用 git checkout 恢复原始版本")
        print()


if __name__ == "__main__":
    main()
