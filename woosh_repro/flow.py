"""最小 Woosh-Flow/VFlow/DFlow/DVFlow 复现。

该文件实现真实可运行的 latent flow matching、video conditioning、CFG 采样和
distilled FlowMap-style 少步采样，但模型维度远小于官方实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class WooshFlowConfig:
    """小型 Flow/VFlow/DFlow 的共享配置。"""

    latent_dim: int = 34
    cond_dim: int = 64
    video_dim: int = 3
    width: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_audio_tokens: int = 128
    max_cond_tokens: int = 64


@dataclass
class FlowCondition:
    """Flow 模型使用的条件容器。

    Attributes:
        text_tokens: CLAP 文本 token latent，形状 ``[B, T_text, cond_dim]``。
        video_tokens: video conditioner 输出，形状 ``[B, T_video, cond_dim]``。
    """

    text_tokens: Tensor | None = None
    video_tokens: Tensor | None = None


def _time_features(t: Tensor, width: int) -> Tensor:
    """生成 sinusoidal time embedding，输入/输出形状为 ``[B] -> [B, width]``。"""
    half = width // 2
    scale = -torch.log(torch.tensor(10_000.0, device=t.device, dtype=t.dtype))
    freqs = torch.exp(torch.linspace(0, 1, half, device=t.device, dtype=t.dtype) * scale)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < width:
        emb = F.pad(emb, (0, width - emb.shape[-1]))
    return emb


class TinyJointTransformer(nn.Module):
    """把条件 token 与 audio latent token 拼接后预测 flow velocity。

    Audio latent 输入为 ``[B, C, F]``，内部转为 ``[B, F, width]``；text/video
    condition 被投影到同一 hidden size 后作为前缀 token。输出 velocity 与输入
    latent 同形状。
    """

    def __init__(self, config: WooshFlowConfig, *, use_video: bool, distilled: bool) -> None:
        super().__init__()
        self.config = config
        self.use_video = use_video
        self.distilled = distilled
        self.audio_in = nn.Linear(config.latent_dim, config.width)
        self.text_in = nn.Linear(config.cond_dim, config.width)
        self.video_in = nn.Linear(config.cond_dim, config.width)
        self.time_in = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.return_time_in = nn.Sequential(
            nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        self.audio_pos = nn.Parameter(torch.randn(1, config.max_audio_tokens, config.width) * 0.01)
        self.cond_pos = nn.Parameter(torch.randn(1, config.max_cond_tokens, config.width) * 0.01)
        self.audio_type = nn.Parameter(torch.zeros(1, 1, config.width))
        self.text_type = nn.Parameter(torch.randn(1, 1, config.width) * 0.01)
        self.video_type = nn.Parameter(torch.randn(1, 1, config.width) * 0.01)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.num_heads,
                dim_feedforward=config.width * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )
        self.audio_out = nn.Linear(config.width, config.latent_dim)

    def _project_condition(self, cond: FlowCondition | None, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """投影 text/video 条件；无条件时返回单个零 token。"""
        chunks: list[Tensor] = []
        if cond is not None and cond.text_tokens is not None:
            text = cond.text_tokens.to(device=device, dtype=dtype)
            chunks.append(self.text_in(text) + self.text_type)
        if self.use_video and cond is not None and cond.video_tokens is not None:
            video = cond.video_tokens.to(device=device, dtype=dtype)
            chunks.append(self.video_in(video) + self.video_type)
        if not chunks:
            return torch.zeros(batch, 1, self.config.width, device=device, dtype=dtype)
        tokens = torch.cat(chunks, dim=1)
        return tokens + self.cond_pos[:, : tokens.shape[1]].to(device=device, dtype=dtype)

    def forward(self, x: Tensor, t: Tensor, cond: FlowCondition | None = None, r: Tensor | None = None) -> Tensor:
        """预测 velocity。

        Args:
            x: latent ``[B, latent_dim, frames]``。
            t: flow matching 时间 ``[B]``。
            cond: 可选 text/video 条件。
            r: distilled/FlowMap 路径使用的第二个时间 ``[B]``。
        """
        if x.ndim != 3:
            raise ValueError("latent tensor must have shape [batch, channels, frames]")
        batch, _, frames = x.shape
        audio = x.transpose(1, 2)
        audio_tokens = self.audio_in(audio)
        audio_tokens = audio_tokens + self.audio_pos[:, :frames].to(x.device, x.dtype) + self.audio_type
        time = self.time_in(_time_features(t.to(dtype=x.dtype), self.config.width)).unsqueeze(1)
        audio_tokens = audio_tokens + time
        if self.distilled and r is not None:
            ret = self.return_time_in(_time_features(r.to(dtype=x.dtype), self.config.width)).unsqueeze(1)
            audio_tokens = audio_tokens + ret
        cond_tokens = self._project_condition(cond, batch, x.device, x.dtype)
        tokens = torch.cat([cond_tokens, audio_tokens], dim=1)
        encoded = self.transformer(tokens)
        audio_encoded = encoded[:, cond_tokens.shape[1] :]
        return self.audio_out(audio_encoded).transpose(1, 2)


class VideoConditioner(nn.Module):
    """从 synthetic video 中提取小型帧级条件 token。

    输入形状为 ``[B, frames, C, H, W]``。每帧使用通道均值和亮度质心
    ``(cy, cx)`` 构造特征，输出 ``[B, frames, cond_dim]``。
    """

    def __init__(self, in_channels: int = 3, cond_dim: int = 64) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(in_channels + 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, video: Tensor) -> Tensor:
        """编码 video 条件 token。"""
        if video.ndim != 5:
            raise ValueError("video must have shape [batch, frames, channels, height, width]")
        batch, frames, channels, height, width = video.shape
        pooled = video.mean(dim=(-1, -2))
        ys = torch.linspace(-1, 1, height, device=video.device, dtype=video.dtype)
        xs = torch.linspace(-1, 1, width, device=video.device, dtype=video.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        mass = video.mean(dim=2).clamp_min(0)
        denom = mass.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        cy = (mass * grid_y).sum(dim=(-1, -2), keepdim=True) / denom
        cx = (mass * grid_x).sum(dim=(-1, -2), keepdim=True) / denom
        features = torch.cat([pooled, cy.view(batch, frames, 1), cx.view(batch, frames, 1)], dim=-1)
        if channels != 3:
            features = features[..., :5]
        return self.frame_encoder(features)


class WooshFlow(nn.Module):
    """最小 text-to-audio flow matching 模型。

    训练时在 ``x0`` 噪声和 ``x1`` 目标 latent 之间线性插值，预测常量速度
    ``x1 - x0``；采样时用 Euler steps 从 noise 推进到生成 latent。
    """

    def __init__(self, config: WooshFlowConfig | None = None, *, use_video: bool = False, distilled: bool = False) -> None:
        super().__init__()
        self.config = config or WooshFlowConfig()
        self.use_video = use_video
        self.distilled = distilled
        self.net = TinyJointTransformer(self.config, use_video=use_video, distilled=distilled)

    def velocity(self, x: Tensor, t: Tensor, cond: FlowCondition | None = None, r: Tensor | None = None) -> Tensor:
        """返回与 ``x`` 同形状的 velocity 预测。"""
        return self.net(x, t, cond, r=r)

    def guided_velocity(self, x: Tensor, t: Tensor, cond: FlowCondition | None, cfg_scale: float = 1.0) -> Tensor:
        """用 classifier-free guidance 合成有条件与无条件 velocity。"""
        if cfg_scale == 1.0 or cond is None:
            return self.velocity(x, t, cond)
        cond_v = self.velocity(x, t, cond)
        uncond_v = self.velocity(x, t, None)
        return uncond_v + cfg_scale * (cond_v - uncond_v)

    def training_loss(self, x1: Tensor, cond: FlowCondition | None = None, noise: Tensor | None = None) -> dict[str, Tensor]:
        """计算基础 flow matching MSE 损失。

        Args:
            x1: 目标 AE latent，形状 ``[B, latent_dim, frames]``。
            cond: 可选条件。
            noise: 可选固定起点 ``x0``，用于可重复测试。
        """
        x0 = torch.randn_like(x1) if noise is None else noise
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t_view = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target = x1 - x0
        pred = self.velocity(xt, t, cond)
        loss = F.mse_loss(pred, target)
        return {"loss": loss, "velocity_mse": loss, "pred_velocity": pred, "target_velocity": target}

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        cond: FlowCondition | None = None,
        *,
        steps: int = 8,
        cfg_scale: float = 1.0,
        device: torch.device | str = "cpu",
        noise: Tensor | None = None,
    ) -> Tensor:
        """用少量 Euler steps 生成 latent，返回形状等于 ``shape`` 的 Tensor。"""
        device = torch.device(device)
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=device, dtype=x.dtype)
            x = x + (1.0 / steps) * self.guided_velocity(x, t, cond, cfg_scale=cfg_scale)
        return x


class WooshVFlow(WooshFlow):
    """带 video token 条件的最小 Woosh-VFlow。"""

    def __init__(self, config: WooshFlowConfig | None = None) -> None:
        super().__init__(config, use_video=True, distilled=False)
        self.video_conditioner = VideoConditioner(cond_dim=self.config.cond_dim)

    def condition(self, text_tokens: Tensor | None, video: Tensor | None) -> FlowCondition:
        """组合 text token 与 video conditioner 输出。"""
        video_tokens = self.video_conditioner(video) if video is not None else None
        return FlowCondition(text_tokens=text_tokens, video_tokens=video_tokens)


class WooshDFlow(WooshFlow):
    """最小 distilled FlowMap/MeanFlow-style text-to-audio 模型。"""

    def __init__(self, config: WooshFlowConfig | None = None) -> None:
        super().__init__(config, use_video=False, distilled=True)

    def training_loss(
        self,
        x1: Tensor,
        cond: FlowCondition | None = None,
        teacher: WooshFlow | None = None,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """计算蒸馏损失。

        若提供 teacher，则目标 velocity 来自 teacher；否则退化为基础
        ``x1 - x0`` 目标。``r`` 是 distilled sampler 的第二个时间步。
        """
        x0 = torch.randn_like(x1) if noise is None else noise
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        r = t * torch.rand_like(t)
        t_view = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        with torch.no_grad():
            target = teacher.velocity(xt, t, cond) if teacher is not None else x1 - x0
        pred = self.velocity(xt, t, cond, r=r)
        loss = F.mse_loss(pred, target)
        return {"loss": loss, "meanflow_mse": loss, "pred_velocity": pred, "target_velocity": target}

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        cond: FlowCondition | None = None,
        *,
        steps: int = 4,
        cfg_scale: float = 1.0,
        renoise: tuple[float, ...] = (0.0, 0.5, 0.5, 0.3),
        device: torch.device | str = "cpu",
        noise: Tensor | None = None,
    ) -> Tensor:
        """执行少步 distilled 采样，并可按 schedule 注入 renoise。"""
        device = torch.device(device)
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        for i in range(steps):
            t_value = i / steps
            r_value = min(1.0, (i + 1) / steps)
            t = torch.full((shape[0],), t_value, device=device, dtype=x.dtype)
            r = torch.full_like(t, r_value)
            v = self.velocity(x, t, cond, r=r)
            if cfg_scale != 1.0 and cond is not None:
                uncond = self.velocity(x, t, None, r=r)
                v = uncond + cfg_scale * (v - uncond)
            x = x + (r_value - t_value) * v
            weight = renoise[i] if i < len(renoise) else 0.0
            if weight:
                x = (1.0 - weight) * x + weight * torch.randn_like(x)
        return x


class WooshDVFlow(WooshDFlow):
    """带 video 条件的最小 distilled VFlow。"""

    def __init__(self, config: WooshFlowConfig | None = None) -> None:
        WooshFlow.__init__(self, config, use_video=True, distilled=True)
        self.video_conditioner = VideoConditioner(cond_dim=self.config.cond_dim)

    def condition(self, text_tokens: Tensor | None, video: Tensor | None) -> FlowCondition:
        """组合 text/video 条件，供 distilled video sampler 使用。"""
        video_tokens = self.video_conditioner(video) if video is not None else None
        return FlowCondition(text_tokens=text_tokens, video_tokens=video_tokens)
