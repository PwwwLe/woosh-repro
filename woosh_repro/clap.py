"""最小 Woosh-CLAP 复现。

用稳定 hash tokenizer、小型文本 Transformer 和小型音频 Transformer 实现
CLAP-style contrastive forward/loss，并导出文本 token latent 供 Flow 条件使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class WooshCLAPConfig:
    """轻量 Woosh-CLAP 配置。

    文本和音频 encoder 都使用小型 Transformer。``embed_dim`` 是 contrastive
    shared space 维度，``width`` 是 token hidden size。
    """

    vocab_size: int = 2048
    max_tokens: int = 16
    audio_n_fft: int = 64
    audio_hop_length: int = 32
    width: int = 64
    embed_dim: int = 32
    temperature: float = 0.2


def _stable_hash_token(token: str, vocab_size: int) -> int:
    """把文本 token 稳定映射到非零词表 id，避免 Python hash 随进程变化。"""
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % (vocab_size - 1) + 1


class WooshCLAP(nn.Module):
    """小型 CLAP-like text/audio alignment model。

    输入 audio 为 ``[B, 1, T]``，text 为 ``list[str]``。模型输出归一化的
    audio/text embedding ``[B, embed_dim]``、文本 token latent
    ``[B, max_tokens, width]``，以及 ``[B, B]`` 的对比学习 logits。
    """

    def __init__(self, config: WooshCLAPConfig | None = None) -> None:
        super().__init__()
        self.config = config or WooshCLAPConfig()
        self.text_embedding = nn.Embedding(self.config.vocab_size, self.config.width, padding_idx=0)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.width,
                nhead=4,
                dim_feedforward=self.config.width * 2,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.text_head = nn.Linear(self.config.width, self.config.embed_dim)
        self.audio_frame = nn.Linear(self.config.audio_n_fft // 2 + 1, self.config.width)
        self.audio_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.width,
                nhead=4,
                dim_feedforward=self.config.width * 2,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.audio_head = nn.Linear(self.config.width, self.config.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / self.config.temperature).log())
        self.register_buffer(
            "audio_window", torch.hann_window(self.config.audio_n_fft), persistent=False
        )

    def tokenize(self, texts: list[str], device: torch.device | str) -> Tensor:
        """将字符串批次转为固定长度 token ids ``[B, max_tokens]``。"""
        ids = torch.zeros(len(texts), self.config.max_tokens, dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            tokens = text.lower().replace(",", " ").replace(".", " ").split()
            for col, token in enumerate(tokens[: self.config.max_tokens]):
                ids[row, col] = _stable_hash_token(token, self.config.vocab_size)
        return ids

    def encode_text_tokens(self, texts: list[str]) -> Tensor:
        """编码文本 token 序列，返回 ``[B, max_tokens, width]``。"""
        device = self.text_embedding.weight.device
        token_ids = self.tokenize(texts, device=device)
        padding_mask = token_ids.eq(0)
        tokens = self.text_embedding(token_ids)
        return self.text_encoder(tokens, src_key_padding_mask=padding_mask)

    def encode_text(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """返回文本全局 embedding 与 token latent。

        Returns:
            tuple[Tensor, Tensor]: ``([B, embed_dim], [B, max_tokens, width])``。
        """
        tokens = self.encode_text_tokens(texts)
        pooled = tokens.mean(dim=1)
        embedding = F.normalize(self.text_head(pooled), dim=-1)
        return embedding, tokens

    def encode_audio(self, audio: Tensor) -> Tensor:
        """编码单声道 audio 到 shared embedding ``[B, embed_dim]``。"""
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, 1, samples]")
        spec = torch.stft(
            audio[:, 0],
            n_fft=self.config.audio_n_fft,
            hop_length=self.config.audio_hop_length,
            window=self.audio_window.to(audio.device, audio.dtype),
            return_complex=True,
            center=True,
        ).abs()
        frames = torch.log1p(spec).transpose(1, 2)
        tokens = self.audio_frame(frames)
        encoded = self.audio_encoder(tokens)
        pooled = encoded.mean(dim=1)
        return F.normalize(self.audio_head(pooled), dim=-1)

    def forward(self, audio: Tensor, texts: list[str]) -> dict[str, Tensor]:
        """执行 CLAP 前向并返回 embedding、文本 token 与相似度 logits。"""
        text_embedding, text_tokens = self.encode_text(texts)
        audio_embedding = self.encode_audio(audio)
        logits = self.logit_scale.exp() * audio_embedding @ text_embedding.t()
        return {
            "audio_embedding": audio_embedding,
            "text_embedding": text_embedding,
            "text_tokens": text_tokens,
            "logits": logits,
        }

    def contrastive_loss(self, audio: Tensor, texts: list[str]) -> dict[str, Tensor]:
        """计算对称 audio-to-text/text-to-audio cross-entropy 损失。"""
        out = self(audio, texts)
        labels = torch.arange(out["logits"].shape[0], device=out["logits"].device)
        audio_to_text = F.cross_entropy(out["logits"], labels)
        text_to_audio = F.cross_entropy(out["logits"].t(), labels)
        loss = 0.5 * (audio_to_text + text_to_audio)
        out.update({"loss": loss, "audio_to_text": audio_to_text, "text_to_audio": text_to_audio})
        return out
