from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ssl_asr.data.manifests import JsonlSpeechDataset, collate_batch
from ssl_asr.models.conformer_ctc import ConformerCTC, ConformerCTCConfig
from ssl_asr.models.ctc import ctc_loss
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

    model_cfg = ConformerCTCConfig(
        vocab_size=len(vocab) + 1,
        n_mels=int(cfg["baseline"]["n_mels"]),
        d_model=int(cfg["baseline"]["d_model"]),
        n_heads=int(cfg["baseline"]["n_heads"]),
        n_layers=int(cfg["baseline"]["n_layers"]),
        dropout=float(cfg["baseline"]["dropout"]),
        time_mask_param=int(cfg["baseline"].get("time_mask_param", 0)),
        freq_mask_param=int(cfg["baseline"].get("freq_mask_param", 0)),
        num_time_masks=int(cfg["baseline"].get("num_time_masks", 0)),
        num_freq_masks=int(cfg["baseline"].get("num_freq_masks", 0)),
    )
    model = ConformerCTC(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    save_run_metadata(run_dir, config={"stage": "baseline", "run_name": args.run_name, **cfg})

    loop_cfg = TrainLoopConfig(
        max_steps=int(cfg["training"]["max_steps"]),
        log_every=int(cfg["training"]["log_every"]),
        save_every=int(cfg["training"]["save_every"]),
        grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
    )

    def step_fn(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)

        log_probs, out_lens = model(wav, wav_lens)

        # Flatten targets for CTCLoss.
        flat_toks = []
        for i in range(tokens.size(0)):
            flat_toks.append(tokens[i, : tok_lens[i]].contiguous())
        flat = torch.cat(flat_toks, dim=0)

        loss = ctc_loss(log_probs, flat, out_lens, tok_lens)
        return {"loss": loss}

    train_loop(model=model, optimizer=optimizer, loader=loader, device=device, cfg=loop_cfg, run_dir=run_dir, step_fn=step_fn)


if __name__ == "__main__":
    main()
