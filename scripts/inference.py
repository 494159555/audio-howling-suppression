#!/usr/bin/env python3
"""音频啸叫抑制推理脚本

使用训练好的AudioUNet5模型进行音频啸叫抑制
支持Griffin-Lim和ISTFT两种相位重构方法
"""

import os
import sys
from pathlib import Path

import torch
import torchaudio
import argparse

from src.models import AudioUNet5


def inference(model_path: str, input_wav: str, output_wav: str, 
              device: str = "cpu", use_griffin_lim: bool = True) -> bool:
    """执行音频啸叫抑制推理
    
    Args:
        model_path: 模型检查点路径
        input_wav: 输入音频路径
        output_wav: 输出音频路径
        device: 计算设备
        use_griffin_lim: 是否使用Griffin-Lim算法
        
    Returns:
        是否成功
    """
    print(f"🔄 处理中: {input_wav} ...")

    # 1. 模型加载
    model = AudioUNet5().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    model.eval()

    # 2. 音频加载
    try:
        waveform, sr = torchaudio.load(input_wav)
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return False

    # 重采样到16kHz
    if sr != 16000:
        print(f"⚠️ 重采样: {sr}Hz -> 16000Hz")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # 3. 频谱分析参数
    n_fft = 512
    hop_length = 128
    win_length = n_fft
    window = torch.hann_window(n_fft).to(device)

    # 4. 幅度谱提取
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
    ).to(device)

    waveform = waveform.to(device)
    mag = spec_transform(waveform).sqrt()  # [channel, 257, Time]

    # 5. 输入预处理
    eps = 1e-8
    norm_min = -11.5
    norm_max = 2.5

    # 对数变换
    mag_log = torch.log10(mag + eps)
    
    # 归一化到[0, 1]
    mag_norm = (mag_log - norm_min) / (norm_max - norm_min)
    
    # 裁剪频率维度 (257 -> 256)
    input_mag = mag_norm[:, :-1, :]

    # 添加批次维度
    input_tensor = input_mag.unsqueeze(0)  # [1, channel, 256, Time]

    # 6. 时间维度填充
    original_len = input_tensor.shape[-1]
    pad_len = 0
    if original_len % 16 != 0:
        pad_len = 16 - (original_len % 16)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_len))

    # 7. 模型推理
    print("🧠 模型推理中...")
    with torch.no_grad():
        pred_norm = model(input_tensor)

    # 8. 输出后处理
    # 移除批次维度和填充
    pred_norm = pred_norm.squeeze(0)  # [channel, 256, Time_padded]
    if pad_len > 0:
        pred_norm = pred_norm[..., :original_len]

    # 反归一化和逆对数变换
    pred_log = pred_norm * (norm_max - norm_min) + norm_min
    pred_linear = torch.pow(10, pred_log)

    # 恢复频率维度 (256 -> 257)
    padding_freq = torch.zeros(pred_linear.shape[0], 1, pred_linear.shape[2]).to(device)
    pred_linear = torch.cat([pred_linear, padding_freq], dim=1)

    # 9. 相位重构和波形合成
    if use_griffin_lim:
        print("✨ 使用Griffin-Lim重构相位...")
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=32,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
        ).to(device)

        new_waveform = griffin_lim(pred_linear)
    else:
        print("⚠️ 使用原始相位重构 (ISTFT)...")
        # 计算原始相位
        stft_complex = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        phase = torch.angle(stft_complex)

        # 重构复频谱
        new_stft_complex = pred_linear * torch.exp(1j * phase)

        # 逆STFT
        new_waveform = torch.istft(
            new_stft_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=waveform.shape[1],
        )

    # 10. 智能归一化
    max_val = torch.max(torch.abs(new_waveform))
    if max_val > 1.0:
        new_waveform = new_waveform / max_val
        print(f"🔊 检测到幅度溢出 ({max_val:.2f})，已自动限制")

    # 11. 保存输出
    try:
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        torchaudio.save(output_wav, new_waveform.cpu(), sr)
        print(f"✅ 处理完成! 保存至: {output_wav}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def main() -> None:
    """命令行接口"""
    parser = argparse.ArgumentParser(description="音频啸叫抑制推理脚本")
    parser.add_argument("--model", type=str, required=True, 
                       help="模型检查点路径")
    parser.add_argument("--input", type=str, required=True, 
                       help="输入音频路径")
    parser.add_argument("--output", type=str, required=True, 
                       help="输出音频路径")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], 
                       help="计算设备")
    parser.add_argument("--use_griffin_lim", action="store_true", default=True,
                       help="使用Griffin-Lim算法重构相位")
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🎯 使用设备: {device}")
    
    # 输入验证
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ 输入音频不存在: {args.input}")
        return
    
    # 执行推理
    success = inference(
        model_path=args.model,
        input_wav=args.input,
        output_wav=args.output,
        device=device,
        use_griffin_lim=args.use_griffin_lim
    )
    
    if success:
        print("🎉 推理成功!")
    else:
        print("💥 推理失败!")


if __name__ == "__main__":
    main()