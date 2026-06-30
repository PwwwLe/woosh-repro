"""最小 Woosh-AE 复现。

实现轻量 STFT/iSTFT latent autoencoder，供 smoke tests 验证重建、loss 和
反向传播。该文件不依赖官方权重或外部数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class WooshAEConfig:
    """WooshAE 的小型 STFT 配置。

    Attributes:
        sample_rate: synthetic audio 的采样率，仅用于示例标注。
        n_fft: STFT/iSTFT 的 FFT 大小。
        hop_length: STFT 帧移，决定 latent 的时间帧数。
        latent_dim: 1x1 投影后的 latent channel 数；None 表示等于复系数维度。
        spectral_windows: 多尺度谱重建损失使用的窗口长度。
    """

    sample_rate: int = 16_000
    n_fft: int = 32
    hop_length: int = 16
    latent_dim: int | None = None
    spectral_windows: tuple[int, ...] = (16, 32, 64)


class WooshAE(nn.Module):
    """小型 STFT-domain Woosh-AE analogue。

    官方 Woosh-AE 是大型 VOCOS-style encoder/decoder。本复现保留同样的
    “waveform -> STFT coefficients -> latent -> iSTFT waveform”数据流：
    输入 audio 为 ``[B, 1, T]``，输出 latent 为 ``[B, latent_dim, F]``。
    当 ``latent_dim`` 等于复系数维度时，1x1 encoder/decoder 以 identity
    初始化，因此初始重建路径接近精确 STFT/iSTFT。
    """

    def __init__(self, config: WooshAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or WooshAEConfig()
        self.freq_bins = self.config.n_fft // 2 + 1
        self.coeff_dim = self.freq_bins * 2
        latent_dim = self.config.latent_dim or self.coeff_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Conv1d(self.coeff_dim, latent_dim, kernel_size=1, bias=False)
        self.decoder = nn.Conv1d(latent_dim, self.coeff_dim, kernel_size=1, bias=False)
        self.register_buffer("window", torch.hann_window(self.config.n_fft), persistent=False)
        self._init_identity()

    def _init_identity(self) -> None:
        """将 1x1 encoder/decoder 初始化为共享通道上的恒等映射。"""
        with torch.no_grad():
            self.encoder.weight.zero_()
            self.decoder.weight.zero_()
            shared = min(self.coeff_dim, self.latent_dim)
            for i in range(shared):
                self.encoder.weight[i, i, 0] = 1.0
                self.decoder.weight[i, i, 0] = 1.0

    def _stft(self, audio: Tensor) -> Tensor:
        """把 ``[B, 1, T]`` waveform 转为 complex STFT ``[B, bins, frames]``。"""
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, 1, samples]")
        return torch.stft(
            audio[:, 0],
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.window.to(audio.device, audio.dtype),
            return_complex=True,
            center=True,
        )

    def _istft(self, coeffs: Tensor, length: int) -> Tensor:
        """把 complex STFT ``[B, bins, frames]`` 还原为 ``[B, 1, length]``。"""
        audio = torch.istft(
            coeffs,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.window.to(coeffs.device, coeffs.real.dtype),
            length=length,
            center=True,
        )
        return audio.unsqueeze(1)

    def encode(self, audio: Tensor) -> Tensor:
        """编码 waveform 到 latent。

        Args:
            audio: 单声道 waveform，形状 ``[B, 1, T]``。

        Returns:
            Tensor: latent 序列，形状 ``[B, latent_dim, frames]``。
        """
        stft = self._stft(audio)
        coeffs = torch.cat([stft.real, stft.imag], dim=1)
        return self.encoder(coeffs)

    def decode(self, latents: Tensor, length: int) -> Tensor:
        """从 latent 解码 waveform。

        Args:
            latents: 形状 ``[B, latent_dim, frames]`` 的 latent。
            length: 输出 waveform 的目标采样点数。

        Returns:
            Tensor: 单声道 waveform，形状 ``[B, 1, length]``。
        """
        coeffs = self.decoder(latents)
        real, imag = coeffs[:, : self.freq_bins], coeffs[:, self.freq_bins :]
        return self._istft(torch.complex(real, imag), length=length)

    def forward(self, audio: Tensor) -> Tensor:
        """等价于 :meth:`encode`，用于把 AE 当作 encoder 调用。"""
        return self.encode(audio)

    def inverse(self, latents: Tensor, length: int) -> Tensor:
        """等价于 :meth:`decode`，与官方组件的 ``inverse`` 命名保持一致。"""
        return self.decode(latents, length=length)

    def reconstruct(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """执行 encode/decode 重建。

        Returns:
            tuple[Tensor, Tensor]: ``(latents, reconstructed_audio)``。
        """
        latents = self.encode(audio)
        return latents, self.decode(latents, length=audio.shape[-1])

    def training_loss(self, audio: Tensor) -> dict[str, Tensor]:
        """计算 AE 主训练损失。

        损失由 waveform L1 与多尺度 log-magnitude STFT L1 组成；所有项都是真
        实前向计算产生的 Tensor，可直接反向传播。
        """
        _, reconstructed = self.reconstruct(audio)
        waveform = F.l1_loss(reconstructed, audio)
        spectral = torch.zeros((), device=audio.device, dtype=audio.dtype)
        for win in self.config.spectral_windows:
            if win > audio.shape[-1]:
                continue
            hop = max(1, win // 4)
            window = torch.hann_window(win, device=audio.device, dtype=audio.dtype)
            target = torch.stft(audio[:, 0], win, hop_length=hop, window=window, return_complex=True)
            pred = torch.stft(reconstructed[:, 0], win, hop_length=hop, window=window, return_complex=True)
            spectral = spectral + F.l1_loss(torch.log1p(pred.abs()), torch.log1p(target.abs()))
        total = waveform + spectral
        return {"loss": total, "waveform_l1": waveform, "spectral_l1": spectral}
