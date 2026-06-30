"""最小复现使用的设备选择、synthetic data 和断言工具。"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pick_device() -> torch.device:
    """选择当前进程可用的单设备；优先 ``cuda:0``，否则 CPU。"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def synthetic_audio(
    batch_size: int = 2,
    length: int = 512,
    sample_rate: int = 16_000,
    device: torch.device | str = "cpu",
) -> Tensor:
    """创建带 transient 的小型单声道 synthetic audio。

    Returns:
        Tensor: waveform，形状 ``[batch_size, 1, length]``，数值裁剪到
        ``[-1, 1]``。
    """
    device = torch.device(device)
    t = torch.linspace(0, 1, length, device=device).view(1, 1, length)
    freqs = torch.linspace(180, 780, batch_size, device=device).view(batch_size, 1, 1)
    chirp = torch.sin(2 * math.pi * (freqs * t + 0.5 * freqs * t.square()))
    burst_center = torch.linspace(0.18, 0.72, batch_size, device=device).view(batch_size, 1, 1)
    burst = torch.exp(-((t - burst_center) ** 2) / 0.0015)
    click_idx = torch.linspace(length // 5, length - length // 5, batch_size, device=device).long()
    clicks = torch.zeros(batch_size, 1, length, device=device)
    clicks[torch.arange(batch_size, device=device), 0, click_idx] = 1.0
    audio = 0.22 * chirp + 0.35 * burst * torch.sin(2 * math.pi * 2600 * t) + 0.45 * clicks
    return audio.clamp(-1, 1)


def synthetic_video(
    batch_size: int = 2,
    frames: int = 4,
    height: int = 16,
    width: int = 16,
    device: torch.device | str = "cpu",
) -> Tensor:
    """创建带移动亮块的 synthetic video。

    Returns:
        Tensor: 视频张量，形状 ``[B, F, C, H, W]``，用于 VFlow 条件测试。
    """
    device = torch.device(device)
    video = torch.zeros(batch_size, frames, 3, height, width, device=device)
    for b in range(batch_size):
        for f in range(frames):
            row = (2 + b * 3 + f * 2) % max(1, height - 3)
            col = (1 + b * 5 + f * 3) % max(1, width - 3)
            video[b, f, :, row : row + 3, col : col + 3] = torch.tensor(
                [1.0, 0.45 + 0.1 * b, 0.2 + 0.1 * f], device=device
            ).view(3, 1, 1)
    return video


def count_parameters(module: torch.nn.Module) -> int:
    """统计可训练参数数量。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_finite(tensor: Tensor, name: str) -> None:
    """断言 Tensor 中没有 NaN 或 Inf。"""
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")
