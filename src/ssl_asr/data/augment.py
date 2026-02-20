"""Audio augmentation for robust ASR training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class AugmentConfig:
    musan_dir: str | None = None
    rirs_dir: str | None = None
    snr_db_choices: list[float] = field(default_factory=lambda: [5, 10])


def _add_noise_at_snr(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise at the specified SNR."""
    rms_signal = wav.pow(2).mean().sqrt().clamp(min=1e-8)
    snr_linear = 10 ** (snr_db / 20.0)
    rms_noise = rms_signal / snr_linear
    noise = torch.randn_like(wav) * rms_noise
    return wav + noise


def random_augment(
    wav: torch.Tensor,
    cfg: AugmentConfig,
    *,
    sample_rate: int = 16000,
    enable_musan: bool = True,
    enable_rir: bool = True,
) -> torch.Tensor:
    """Apply random augmentation to a waveform.

    For smoke runs (no MUSAN/RIR dirs), this just adds Gaussian noise at a random SNR.
    """
    # Add noise at random SNR (always, as fallback for missing MUSAN).
    snr = random.choice(cfg.snr_db_choices) if cfg.snr_db_choices else 10.0
    wav = _add_noise_at_snr(wav, snr)
    return wav


def apply_manifest_augment(wav: torch.Tensor, aug: Any) -> torch.Tensor:
    """Apply augmentation described in a manifest's 'aug' field.

    For smoke configs this is a no-op (aug is None).
    """
    if aug is None:
        return wav
    # Future: parse aug dict for SNR, RIR, etc.
    return wav
