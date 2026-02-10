from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class TrainLoopConfig:
    max_steps: int
    log_every: int
    save_every: int
    grad_accum_steps: int


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "extra": extra,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def examples_per_second(num_examples: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(num_examples / seconds)


def train_loop(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainLoopConfig,
    run_dir: Path,
    step_fn,
) -> dict[str, Any]:
    model.train()

    step = 0
    seen = 0
    t0 = time.time()

    pbar = tqdm(total=cfg.max_steps, desc="train", ncols=100)
    optimizer.zero_grad(set_to_none=True)

    last_log = {}

    it = iter(loader)
    while step < cfg.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        loss_dict = step_fn(batch)
        loss = loss_dict["loss"] / max(1, cfg.grad_accum_steps)
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        seen += int(batch["wav"].size(0))

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            elapsed = time.time() - t0
            eps = examples_per_second(seen, elapsed)
            max_cuda_mb = 0.0
            if torch.cuda.is_available() and device.type == "cuda":
                max_cuda_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
            last_log = {
                "step": step,
                "loss": float(loss_dict["loss"].detach().cpu().item()),
                "examples_per_sec": eps,
                "max_cuda_mem_mb": max_cuda_mb,
            }
            pbar.set_postfix(loss=f"{last_log['loss']:.4f}", eps=f"{eps:.1f}")

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                extra=last_log,
            )

        pbar.update(1)

    pbar.close()

    # Write a small metrics file for evaluation aggregation.
    metrics_path = run_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(last_log, indent=2) + "\n", encoding="utf-8")
    return last_log
