from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from ssl_asr.utils.io import write_jsonl


def synth_utt(sr: int, seconds: float, *, freq: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds), dtype=np.float32) / float(sr)
    sig = 0.2 * np.sin(2 * math.pi * freq * t)
    sig += noise * rng.standard_normal(sig.shape).astype(np.float32)
    sig = np.clip(sig, -1.0, 1.0)
    return sig.astype(np.float32)


def make_split(out_dir: Path, name: str, n: int, *, sr: int) -> Path:
    rows = []
    split_dir = out_dir / "audio" / name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Tiny vocabulary-consistent transcripts.
    texts = [
        "hello world",
        "a small test",
        "speech recognition",
        "robust asr",
        "self supervised",
        "consistency loss",
    ]

    for i in range(n):
        text = random.choice(texts)
        wav = synth_utt(sr, seconds=1.2 + 0.2 * (i % 3), freq=220.0 + 20.0 * i, noise=0.01, seed=1000 + i)
        wav_path = split_dir / f"{name}_{i:03d}.wav"
        sf.write(str(wav_path), wav, sr)
        rows.append({"audio_path": str(wav_path.resolve()), "text": text, "sample_rate": sr, "duration_s": float(len(wav) / sr)})

    mpath = out_dir / "manifests" / f"{name}.jsonl"
    write_jsonl(mpath, rows)
    return mpath


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)

    random.seed(0)

    make_split(out_dir, "train", 24, sr=args.sr)
    make_split(out_dir, "dev", 8, sr=args.sr)
    make_split(out_dir, "test", 8, sr=args.sr)


if __name__ == "__main__":
    main()
