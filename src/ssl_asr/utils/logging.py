from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "time_utc": now_utc_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        snap["gpu_name"] = torch.cuda.get_device_name(0)
    return snap


def save_run_metadata(run_dir: Path, *, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    def normalize(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=normalize) + "\n", encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env_snapshot(), indent=2) + "\n", encoding="utf-8")
