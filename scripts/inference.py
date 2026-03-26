#!/usr/bin/env python3
"""音频啸叫抑制推理脚本

本脚本使用训练好的深度学习模型对音频进行啸叫抑制处理。

主要功能：
    - 加载训练好的U-Net模型检查点
    - 对输入音频进行预处理（STFT变换、归一化等）
    - 使用模型预测掩码并抑制啸叫
    - 支持两种相位重构方法：Griffin-Lim和ISTFT
    - 自动处理采样率转换和幅度归一化

使用方法：
    python scripts/inference.py --model experiments/exp_xxx/checkpoints/best_model.pth \\
                                --input input.wav \\
                                --output output.wav \\
                                --device cuda \\
                                --use_griffin_lim

作者：音频处理实验室
版本：1.0
"""

import os
import sys
from pathlib import Path

import torch
import torchaudio
import argparse

# 导入模型（可根据需要更改模型类型）
from src.models import AudioUNet5


def inference(model_path: str, input_wav: str, output_wav: str,
              device: str = "cpu", use_griffin_lim: bool = True) -> bool:
    """执行音频啸叫抑制推理

    这是推理的核心函数，执行完整的音频处理流程：
    1. 加载模型
    2. 加载并预处理音频
    3. STFT变换得到频谱图
    4. 归一化并输入模型
    5. 模型预测并后处理
    6. 相位重构并转换回时域
    7. 保存结果

    Args:
        model_path (str): 模型检查点文件路径，.pth格式
        input_wav (str): 输入音频文件路径，.wav格式
        output_wav (str): 输出音频文件路径，.wav格式
        device (str): 计算设备，可选"cpu"或"cuda"
        use_griffin_lim (bool): 是否使用Griffin-Lim算法重构相位
                                True-音质更好但速度较慢
                                False-使用原始相位，速度快但音质稍差

    Returns:
        bool: 推理是否成功，True表示成功，False表示失败

    处理流程说明：
        - 输入音频会自动重采样到16kHz（如果需要）
        - 使用512点FFT、128点跳跃长度进行STFT
        - 对数频谱归一化到[0,1]区间
        - 模型输出乘性掩码，与输入频谱相乘
        - 最后重构相位并转换回时域波形
    """
    print(f"🔄 开始处理: {input_wav}")

    # ============ 第1步：模型加载 ============
    # 创建模型实例并移动到指定设备
    model = AudioUNet5().to(device)

    try:
        # 加载训练好的模型权重
        # map_location确保加载到正确的设备
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    # 设置为评估模式（关闭dropout、batchnorm等）
    model.eval()

    # ============ 第2步：音频加载 ============
    try:
        # torchaudio.load返回 (waveform, sample_rate)
        # waveform形状: [channels, samples]
        waveform, sr = torchaudio.load(input_wav)
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return False

    # 检查采样率，如果不匹配16kHz则重采样
    # 模型是在16kHz采样率下训练的，所以输入也需要16kHz
    if sr != 16000:
        print(f"⚠️  检测到采样率 {sr}Hz，正在重采样到16000Hz...")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000
        print(f"✅ 重采样完成")

    # ============ 第3步：STFT参数设置 ============
    # 这些参数必须与训练时保持一致
    n_fft = 512              # FFT窗口大小（帧长）
    hop_length = 128         # 帧移（相邻帧之间的重叠）
    win_length = n_fft       # 窗函数长度（通常等于FFT大小）
    window = torch.hann_window(n_fft).to(device)  # 汉宁窗

    # ============ 第4步：幅度谱提取 ============
    # 创建STFT变换器
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,  # 返回功率谱（幅度平方）
    ).to(device)

    # 将波形移动到GPU/CPU
    waveform = waveform.to(device)

    # 计算功率谱并开平方得到幅度谱
    # 输出形状: [channels, freq_bins, time_frames]
    # 对于16kHz、512 FFT: freq_bins = 513 (0~256Hz的频率，共257个正频率)
    mag = spec_transform(waveform).sqrt()  # [channel, 257, Time]

    # ============ 第5步：输入预处理 ============
    # 预处理参数必须与训练时完全一致
    eps = 1e-8           # 防止log(0)的小常数
    norm_min = -11.5     # 对数域最小值（用于归一化）
    norm_max = 2.5       # 对数域最大值（用于归一化）

    # 对数变换：将线性幅度转为对数幅度
    # 公式：log_mag = log10(magnitude + eps)
    mag_log = torch.log10(mag + eps)

    # 归一化到[0, 1]区间
    # 公式：normalized = (log_mag - min) / (max - min)
    mag_norm = (mag_log - norm_min) / (norm_max - norm_min)

    # 裁剪频率维度：257 -> 256
    # 模型输入是256个频率bin，去掉最高频的bin
    input_mag = mag_norm[:, :-1, :]  # [channel, 256, Time]

    # 添加批次维度
    # 模型期望输入形状: [batch, channels, freq, time]
    input_tensor = input_mag.unsqueeze(0)  # [1, channel, 256, Time]

    # ============ 第6步：时间维度填充 ============
    # 某些模型架构要求时间维度能被16整除（因为下采样4次，每次除以2）
    original_len = input_tensor.shape[-1]
    pad_len = 0

    if original_len % 16 != 0:
        pad_len = 16 - (original_len % 16)
        # 在右侧填充0
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_len))
        print(f"📏 时间维度填充: {original_len} -> {original_len + pad_len}")

    # ============ 第7步：模型推理 ============
    print("🧠 正在执行模型推理...")
    with torch.no_grad():  # 不计算梯度，节省内存
        # 模型输出预测的干净音频频谱（归一化后）
        pred_norm = model(input_tensor)

    # ============ 第8步：输出后处理 ============
    # 移除批次维度
    pred_norm = pred_norm.squeeze(0)  # [channel, 256, Time_padded]

    # 移除时间维度的填充
    if pad_len > 0:
        pred_norm = pred_norm[..., :original_len]

    # 反归一化：从[0,1]恢复到对数域
    # 公式：log_mag = normalized * (max - min) + min
    pred_log = pred_norm * (norm_max - norm_min) + norm_min

    # 逆对数变换：从对数域恢复到线性幅度域
    # 公式：magnitude = 10^log_mag
    pred_linear = torch.pow(10, pred_log)

    # 恢复频率维度：256 -> 257
    # 在最高频处补0（恢复被裁剪的频率bin）
    padding_freq = torch.zeros(pred_linear.shape[0], 1, pred_linear.shape[2]).to(device)
    pred_linear = torch.cat([pred_linear, padding_freq], dim=1)

    # ============ 第9步：相位重构和波形合成 ============
    if use_griffin_lim:
        # Griffin-Lum算法：仅从幅度谱迭代估计相位
        # 优点：重构质量高，缺点：计算量大
        print("✨ 使用Griffin-Lim算法重构相位（高质量，速度较慢）...")
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=32,        # 迭代次数，越多质量越好但越慢
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,        # 使用幅度谱而非功率谱
        ).to(device)

        # 从幅度谱重构波形
        new_waveform = griffin_lim(pred_linear)
    else:
        # ISTFT方法：使用原始噪声音频的相位
        # 优点：速度快，缺点：保留了部分噪声相位
        print("⚠️  使用ISTFT方法（速度较快，但保留了原始相位）...")

        # 计算原始音频的STFT以获取相位
        stft_complex = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,  # 返回复数形式
        )

        # 提取相位信息（角度）
        phase = torch.angle(stft_complex)

        # 用预测的幅度和原始相位重构复频谱
        # 复数 = 幅度 * e^(j*相位)
        new_stft_complex = pred_linear * torch.exp(1j * phase)

        # 逆STFT转换回时域
        new_waveform = torch.istft(
            new_stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=waveform.shape[1],  # 确保输出长度一致
        )

    # ============ 第10步：智能归一化 ============
    # 检查是否有幅度溢出（超过1.0会导致削波失真）
    max_val = torch.max(torch.abs(new_waveform))
    if max_val > 1.0:
        new_waveform = new_waveform / max_val
        print(f"🔊 检测到幅度溢出 ({max_val:.2f})，已自动归一化")

    # ============ 第11步：保存输出 ============
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_wav)
        if output_dir:  # 如果指定了目录
            os.makedirs(output_dir, exist_ok=True)

        # 保存为WAV文件
        torchaudio.save(output_wav, new_waveform.cpu(), sr)
        print(f"✅ 处理完成！结果已保存至: {output_wav}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def main() -> None:
    """命令行接口主函数

    解析命令行参数并调用inference函数执行推理。

    使用示例：
        # 使用GPU和Griffin-Lim
        python scripts/inference.py --model checkpoints/best_model.pth \\
                                    --input noisy.wav \\
                                    --output clean.wav \\
                                    --device cuda \\
                                    --use_griffin_lim

        # 使用CPU和ISTFT（快速模式）
        python scripts/inference.py --model checkpoints/best_model.pth \\
                                    --input noisy.wav \\
                                    --output clean.wav \\
                                    --device cpu
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="音频啸叫抑制推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # GPU推理（推荐）
  python scripts/inference.py --model exp/best.pth --input in.wav --output out.wav --device cuda

  # CPU推理
  python scripts/inference.py --model exp/best.pth --input in.wav --output out.wav --device cpu

  # 使用高质量相位重构
  python scripts/inference.py --model exp/best.pth --input in.wav --output out.wav --use_griffin_lim
        """
    )

    # 添加命令行参数
    parser.add_argument("--model", type=str, required=True,
                       help="模型检查点路径 (.pth文件)")
    parser.add_argument("--input", type=str, required=True,
                       help="输入音频文件路径 (.wav文件)")
    parser.add_argument("--output", type=str, required=True,
                       help="输出音频文件路径 (.wav文件)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="计算设备：auto(自动检测)、cpu、cuda（默认：auto）")
    parser.add_argument("--use_griffin_lim", action="store_true", default=True,
                       help="使用Griffin-Lim算法重构相位（默认启用，音质更好）")

    # 解析参数
    args = parser.parse_args()

    # ============ 设备选择 ============
    if args.device == "auto":
        # 自动检测：如果有CUDA GPU则使用，否则使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"🎯 检测到CUDA GPU，使用GPU加速")
        else:
            print(f"🎯 未检测到CUDA GPU，使用CPU")
    else:
        device = torch.device(args.device)

    print(f"🎯 使用设备: {device}")

    # ============ 输入验证 ============
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误：模型文件不存在: {args.model}")
        print(f"   请检查路径是否正确")
        return

    # 检查输入音频是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误：输入音频不存在: {args.input}")
        print(f"   请检查路径是否正确")
        return

    # ============ 执行推理 ============
    print(f"\n{'='*60}")
    print(f"🚀 音频啸叫抑制推理")
    print(f"{'='*60}")
    print(f"模型: {args.model}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"设备: {device}")
    print(f"相位重构: {'Griffin-Lim' if args.use_griffin_lim else 'ISTFT'}")
    print(f"{'='*60}\n")

    success = inference(
        model_path=args.model,
        input_wav=args.input,
        output_wav=args.output,
        device=device,
        use_griffin_lim=args.use_griffin_lim
    )

    # ============ 结果反馈 ============
    print(f"\n{'='*60}")
    if success:
        print("🎉 推理成功！")
        print(f"✅ 处理后的音频已保存到: {args.output}")
        print(f"\n💡 提示：您可以对比输入和输出音频来评估效果")
    else:
        print("💥 推理失败！")
        print(f"   请检查上述错误信息并重试")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()