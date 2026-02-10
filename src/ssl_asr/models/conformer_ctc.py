from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * torch.sigmoid(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = self.layer_norm(x)
        y = y.transpose(1, 2)  # (B,D,T)
        y = self.pointwise_conv1(y)
        y = self.glu(y)
        y = self.depthwise_conv(y)
        y = self.batch_norm(y)
        y = self.swish(y)
        y = self.pointwise_conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return y


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, conv_kernel: int = 31):
        super().__init__()
        d_ff = 4 * d_model
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_drop = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # Macaron style FFN
        x = x + 0.5 * self.ff1(x)

        y = self.mha_ln(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.mha_drop(y)

        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_ln(x)


@dataclass(frozen=True)
class ConformerCTCConfig:
    vocab_size: int
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    # SpecAugment (applied only in training mode)
    time_mask_param: int = 0
    freq_mask_param: int = 0
    num_time_masks: int = 0
    num_freq_masks: int = 0


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ConformerCTCConfig):
        super().__init__()
        self.cfg = cfg

        # Frontend: log-mel
        import torchaudio  # lazy import

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=cfg.n_mels,
        )
        self._time_mask = torchaudio.transforms.TimeMasking(time_mask_param=cfg.time_mask_param) if cfg.time_mask_param > 0 else None
        self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg.freq_mask_param) if cfg.freq_mask_param > 0 else None

        self.in_ln = nn.LayerNorm(cfg.n_mels)
        self.in_proj = nn.Linear(cfg.n_mels, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [ConformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ctc_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        # True where padded
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # wav: (B,T)
        x = self.melspec(wav)  # (B, n_mels, frames)
        x = torch.clamp(x, min=1e-10).log()

        # SpecAugment operates on (B, n_mels, frames)
        if self.training:
            if self._time_mask is not None and self.cfg.num_time_masks > 0:
                for _ in range(int(self.cfg.num_time_masks)):
                    x = self._time_mask(x)
            if self._freq_mask is not None and self.cfg.num_freq_masks > 0:
                for _ in range(int(self.cfg.num_freq_masks)):
                    x = self._freq_mask(x)

        x = x.transpose(1, 2)  # (B, frames, n_mels)

        # Approx frame lengths from hop length.
        frame_lens = torch.div(wav_lens + 159, 160, rounding_mode="floor")

        x = self.in_ln(x)
        x = self.in_proj(x)
        x = self.drop(x)

        max_frames = x.size(1)
        kpm = self._make_padding_mask(frame_lens, max_frames)

        for b in self.blocks:
            x = b(x, kpm)

        logits = self.ctc_head(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, frame_lens
