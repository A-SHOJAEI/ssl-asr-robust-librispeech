from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapCI:
    lo: float
    hi: float


def bootstrap_ci(values: list[float], *, num_samples: int = 1000, alpha: float = 0.05, seed: int = 0) -> BootstrapCI:
    if not values:
        return BootstrapCI(lo=float("nan"), hi=float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(num_samples):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(samp) / n)
    stats.sort()
    lo_i = int((alpha / 2) * num_samples)
    hi_i = int((1 - alpha / 2) * num_samples) - 1
    lo_i = max(0, min(num_samples - 1, lo_i))
    hi_i = max(0, min(num_samples - 1, hi_i))
    return BootstrapCI(lo=float(stats[lo_i]), hi=float(stats[hi_i]))
