from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ssl_asr.data.augment import AugmentConfig, random_augment
from ssl_asr.data.manifests import JsonlSpeechDataset, collate_batch
from ssl_asr.models.ctc import ctc_loss
from ssl_asr.models.wav2vec2 import Wav2Vec2CTC, Wav2Vec2Config, Wav2Vec2Model
from ssl_asr.train.common import TrainLoopConfig, load_checkpoint, train_loop
from ssl_asr.utils.config import load_yaml, parse_bool
from ssl_asr.utils.logging import save_run_metadata
from ssl_asr.utils.seed import DeterminismConfig, seed_everything


def symmetric_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    kl_pq = torch.sum(p * (log_p - log_q), dim=-1)
    kl_qp = torch.sum(q * (log_q - log_p), dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--consistency_weight", type=float, default=1.0)
    ap.add_argument("--augment_musan", type=parse_bool, default=True)
    ap.add_argument("--augment_rir", type=parse_bool, default=True)
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
    base_cfg = Wav2Vec2Config(
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

    base = Wav2Vec2Model(base_cfg)
    model = Wav2Vec2CTC(base, vocab_size=len(vocab) + 1).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    if args.init_from:
        ckpt_path = Path(args.init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init_from not found: {ckpt_path}")
        # Load only the base wav2vec2 weights from SSL checkpoint.
        ssl_model = Wav2Vec2Model(base_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ssl_model.load_state_dict(ckpt["model"], strict=True)
        model.base.load_state_dict(ssl_model.state_dict(), strict=True)

    aug_cfg = AugmentConfig(
        musan_dir=(str(cfg["augment"].get("musan_dir")) or None),
        rirs_dir=(str(cfg["augment"].get("rirs_dir")) or None),
        snr_db_choices=[float(x) for x in cfg["augment"].get("snr_db", [0, 5, 10, 20])],
    )

    save_run_metadata(
        run_dir,
        config={
            "stage": "finetune",
            "run_name": args.run_name,
            "init_from": args.init_from,
            "consistency_weight": args.consistency_weight,
            "augment_musan": bool(args.augment_musan),
            "augment_rir": bool(args.augment_rir),
            **cfg,
        },
    )

    loop_cfg = TrainLoopConfig(
        max_steps=int(cfg["training"]["max_steps"]),
        log_every=int(cfg["training"]["log_every"]),
        save_every=int(cfg["training"]["save_every"]),
        grad_accum_steps=int(cfg["training"]["grad_accum_steps"]),
    )

    def _pad_stack(wavs: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(int(w.numel()) for w in wavs)
        out = torch.zeros((len(wavs), max_len), dtype=torch.float32, device=device)
        for i, w in enumerate(wavs):
            out[i, : w.numel()] = w.to(device)
        return out

    def step_fn(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        wav = batch["wav"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)

        # Two independently augmented views.
        wav1_list = [
            random_augment(
                wav[i, : wav_lens[i]].detach().cpu(),
                aug_cfg,
                sample_rate=int(cfg["data"]["sample_rate"]),
                enable_musan=bool(args.augment_musan),
                enable_rir=bool(args.augment_rir),
            )
            for i in range(wav.size(0))
        ]
        wav2_list = [
            random_augment(
                wav[i, : wav_lens[i]].detach().cpu(),
                aug_cfg,
                sample_rate=int(cfg["data"]["sample_rate"]),
                enable_musan=bool(args.augment_musan),
                enable_rir=bool(args.augment_rir),
            )
            for i in range(wav.size(0))
        ]

        wav1 = _pad_stack(wav1_list)
        wav2 = _pad_stack(wav2_list)

        logp1, out_lens1 = model(wav1, wav_lens)
        logp2, out_lens2 = model(wav2, wav_lens)

        # CTC targets flattened
        flat_toks = []
        for i in range(tokens.size(0)):
            flat_toks.append(tokens[i, : tok_lens[i]].contiguous())
        flat = torch.cat(flat_toks, dim=0)

        ctc = ctc_loss(logp1, flat, out_lens1, tok_lens)

        # Consistency loss: align by minimum time steps per sample.
        min_t = min(logp1.size(1), logp2.size(1))
        cons = symmetric_kl(logp1[:, :min_t, :], logp2[:, :min_t, :]).mean()

        loss = ctc + float(args.consistency_weight) * cons
        return {"loss": loss, "ctc": ctc.detach(), "cons": cons.detach()}

    train_loop(model=model, optimizer=optimizer, loader=loader, device=device, cfg=loop_cfg, run_dir=run_dir, step_fn=step_fn)


if __name__ == "__main__":
    main()
