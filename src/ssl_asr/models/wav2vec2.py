from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn


class ConvFeatureEncoder(nn.Module):
    def __init__(self, channels: list[int], kernel: list[int], stride: list[int]):
        super().__init__()
        if not (len(channels) == len(kernel) == len(stride)):
            raise ValueError("conv_channels/conv_kernel/conv_stride must have same length")
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in zip(channels, kernel, stride, strict=True):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B,T)
        x = wav.unsqueeze(1)  # (B,1,T)
        x = self.net(x)  # (B,C,T')
        return x.transpose(1, 2)  # (B,T',C)


class TransformerContext(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=src_key_padding_mask)


@dataclass(frozen=True)
class Wav2Vec2Config:
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    conv_channels: list[int]
    conv_kernel: list[int]
    conv_stride: list[int]

    mask_prob: float = 0.065
    mask_length: int = 10

    temperature: float = 0.1
    num_negatives: int = 100


class Wav2Vec2Model(nn.Module):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg
        self.feat = ConvFeatureEncoder(cfg.conv_channels, cfg.conv_kernel, cfg.conv_stride)
        feat_dim = cfg.conv_channels[-1]
        self.proj = nn.Linear(feat_dim, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.context = TransformerContext(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)

    def _lengths_after_conv(self, wav_lens: torch.Tensor) -> torch.Tensor:
        # approximate length transform for stride-only convs
        out = wav_lens.clone()
        for s in self.cfg.conv_stride:
            out = torch.div(out + (s - 1), s, rounding_mode="floor")
        return out

    def forward_features(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.feat(wav)  # (B,T',C)
        z = self.proj(z)
        z = self.drop(z)
        out_lens = self._lengths_after_conv(wav_lens)
        return z, out_lens

    def forward_context(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> torch.Tensor:
        max_t = feats.size(1)
        idx = torch.arange(max_t, device=feats.device).unsqueeze(0)
        kpm = idx >= feat_lens.unsqueeze(1)
        return self.context(feats, kpm)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, feat_lens = self.forward_features(wav, wav_lens)
        ctx = self.forward_context(feats, feat_lens)
        return feats, ctx, feat_lens

    def make_mask(self, feat_lens: torch.Tensor) -> torch.Tensor:
        """Create a boolean mask over time steps for each sample."""
        bsz = feat_lens.numel()
        max_t = int(feat_lens.max().item())
        mask = torch.zeros((bsz, max_t), dtype=torch.bool, device=feat_lens.device)
        for b in range(bsz):
            t = int(feat_lens[b].item())
            if t <= 0:
                continue
            num_mask = int(self.cfg.mask_prob * t / max(1, self.cfg.mask_length))
            for _ in range(num_mask):
                start = random.randint(0, max(0, t - self.cfg.mask_length))
                mask[b, start : start + self.cfg.mask_length] = True
        return mask


class Wav2Vec2CTC(nn.Module):
    def __init__(self, base: Wav2Vec2Model, vocab_size: int):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.cfg.d_model, vocab_size)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _feats, ctx, feat_lens = self.base(wav, wav_lens)
        logits = self.head(ctx)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, feat_lens


def info_nce_loss(
    ctx: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
    num_negatives: int,
) -> torch.Tensor:
    """Masked contrastive loss similar to wav2vec2 (without quantization).

    Args:
        ctx: (B,T,D)
        targets: (B,T,D) detached targets
        mask: (B,T) bool (True at masked positions)
    """

    bsz, t, d = ctx.shape
    device = ctx.device

    mask = mask[:, :t]
    masked_idx = mask.nonzero(as_tuple=False)  # (N,2)
    if masked_idx.numel() == 0:
        return torch.zeros((), device=device, dtype=ctx.dtype)

    ctx_m = ctx[masked_idx[:, 0], masked_idx[:, 1]]  # (N,D)
    pos = targets[masked_idx[:, 0], masked_idx[:, 1]].detach()  # (N,D)

    # Sample negatives from the batch/time.
    flat = targets.reshape(bsz * t, d).detach()
    n = masked_idx.size(0)
    neg_ids = torch.randint(0, bsz * t, (n, num_negatives), device=device)
    neg = flat[neg_ids]  # (N,K,D)

    # Similarities
    ctx_m = nn.functional.normalize(ctx_m, dim=-1)
    pos = nn.functional.normalize(pos, dim=-1)
    neg = nn.functional.normalize(neg, dim=-1)

    pos_sim = torch.sum(ctx_m * pos, dim=-1, keepdim=True)  # (N,1)
    neg_sim = torch.einsum("nd,nkd->nk", ctx_m, neg)  # (N,K)

    logits = torch.cat([pos_sim, neg_sim], dim=1) / max(1e-6, temperature)
    labels = torch.zeros((n,), dtype=torch.long, device=device)
    return nn.functional.cross_entropy(logits, labels)
