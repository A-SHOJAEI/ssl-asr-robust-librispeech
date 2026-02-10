from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int
    deterministic: bool = False
    cudnn_benchmark: bool = False


def seed_everything(cfg: DeterminismConfig) -> None:
    """Seed Python, NumPy, and PyTorch.

    Note: full determinism for ASR models can be expensive and may not be
    achievable for all kernels; we expose a switch and log it.
    """

    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    if cfg.deterministic:
        # Raises on some nondeterministic ops.
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
