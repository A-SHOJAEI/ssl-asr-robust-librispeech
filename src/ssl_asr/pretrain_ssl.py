from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ssl_asr.data.manifests import JsonlSpeechDataset, collate_batch
from ssl_asr.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model, info_nce_loss
from ssl_asr.train.common import TrainLoopConfig, train_loop
from ssl_asr.utils.config import load_yaml
from ssl_asr.utils.logging import save_run_metadata
from ssl_asr.utils.seed import DeterminismConfig, seed_everything


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    run_dir = Path("runs") / args.run_name

    seed_everything(DeterminismConfig(seed=int(cfg["seed"]), deterministic=bool(args.deterministic)))

    device = torch.device(str(cfg.get("device", "cpu")))

    vocab = str(cfg["model"]["vocab"])
    ds = JsonlSpeechDataset(cfg["data"]["train_manifest"], target_sr=int(cfg["data"]["sample_rate"]), vocab=vocab)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_batch,
        drop_last=False,
    )

    wcfg = cfg["wav2vec2"]
    model_cfg = Wav2Vec2Config(
        d_model=int(wcfg["d_model"]),
        n_heads=int(wcfg["n_heads"]),
        n_layers=int(wcfg["n_layers"]),
        dropout=float(wcfg["dropout"]),
        conv_channels=[int(x) for x in wcfg["conv_channels"]],
        conv_kernel=[int(x) for x in wcfg["conv_kernel"]],
        conv_stride=[int(x) for x in wcfg["conv_stride"]],
        mask_prob=float(wcfg["mask_prob"]),
        mask_length=int(wcfg["mask_length"]),
        temperature=float(wcfg["temperature"]),
        num_negatives=int(wcfg["num_negatives"]),
    )

    model = Wav2Vec2Model(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    save_run_metadata(run_dir, config={"stage": "ssl_pretrain", "run_name": args.run_name, **cfg})

    loop_cfg = TrainLoopConfig(
        max_steps=int(cfg["training"]["max_steps"]),
        log_every=int(cfg["training"]["log_every"]),
        save_every=int(cfg["training"]["save_every"]),
        grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
    )

    def step_fn(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)

        feats, ctx, feat_lens = model(wav, wav_lens)
        mask = model.make_mask(feat_lens)

        # Targets are the pre-context features (projected conv feats).
        loss = info_nce_loss(
            ctx=ctx,
            targets=feats,
            mask=mask,
            temperature=model_cfg.temperature,
            num_negatives=model_cfg.num_negatives,
        )
        return {"loss": loss}

    train_loop(model=model, optimizer=optimizer, loader=loader, device=device, cfg=loop_cfg, run_dir=run_dir, step_fn=step_fn)


if __name__ == "__main__":
    main()
