from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_asr.data.augment import apply_manifest_augment
from ssl_asr.data.manifests import JsonlSpeechDataset, collate_batch
from ssl_asr.data.text import normalize_text
from ssl_asr.eval.bootstrap import bootstrap_ci
from ssl_asr.eval.wer import compute_wer
from ssl_asr.models.conformer_ctc import ConformerCTC, ConformerCTCConfig
from ssl_asr.models.ctc import greedy_ctc_decode
from ssl_asr.models.wav2vec2 import Wav2Vec2CTC, Wav2Vec2Config, Wav2Vec2Model
from ssl_asr.utils.config import load_yaml


def decode_ids(ids: list[int], vocab: str) -> str:
    itos = {i + 1: c for i, c in enumerate(vocab)}
    s = "".join(itos.get(i, "") for i in ids)
    return normalize_text(s)


def load_model_for_ckpt(ckpt_path: Path, cfg: dict, *, vocab: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Heuristic: baseline conformer checkpoints are saved from ConformerCTC.
    # wav2vec2 finetune checkpoints are saved from Wav2Vec2CTC.
    state = ckpt["model"]

    if any(k.startswith("blocks.") or k.startswith("ctc_head.") for k in state.keys()):
        mcfg = ConformerCTCConfig(
            vocab_size=len(vocab) + 1,
            n_mels=int(cfg["baseline"]["n_mels"]),
            d_model=int(cfg["baseline"]["d_model"]),
            n_heads=int(cfg["baseline"]["n_heads"]),
            n_layers=int(cfg["baseline"]["n_layers"]),
            dropout=float(cfg["baseline"]["dropout"]),
        )
        model = ConformerCTC(mcfg)
        model.load_state_dict(state, strict=True)
        return model

    # wav2vec2
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
    model = Wav2Vec2CTC(base, vocab_size=len(vocab) + 1)
    model.load_state_dict(state, strict=True)
    return model


def evaluate_one(model: torch.nn.Module, ds: JsonlSpeechDataset, *, device: torch.device, vocab: str) -> dict:
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    per_utt_wer: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", ncols=100):
            wav = batch["wav"].to(device)
            wav_lens = batch["wav_lens"].to(device)

            # Apply any manifest-defined augmentation (robust eval manifests).
            aug_list = batch.get("aug")
            if aug_list is not None and any(a for a in aug_list):
                wavs = []
                for i in range(wav.size(0)):
                    w = wav[i, : wav_lens[i]].cpu()
                    w = apply_manifest_augment(w, aug_list[i])
                    wavs.append(w)
                max_len = max(int(w.numel()) for w in wavs)
                wav2 = torch.zeros((len(wavs), max_len), dtype=torch.float32)
                for i, w in enumerate(wavs):
                    wav2[i, : w.numel()] = w
                wav = wav2.to(device)

            logp, out_lens = model(wav, wav_lens)
            pred_ids = greedy_ctc_decode(logp)

            for i in range(len(pred_ids)):
                hyp = decode_ids(pred_ids[i], vocab)
                ref = normalize_text(batch["text"][i])
                refs.append(ref)
                hyps.append(hyp)
                wr = compute_wer([ref], [hyp]).wer
                per_utt_wer.append(float(wr))

    agg = compute_wer(refs, hyps)
    ci = bootstrap_ci(per_utt_wer, num_samples=500, seed=0)
    return {
        "wer": agg.wer,
        "wer_ci": {"lo": ci.lo, "hi": ci.hi},
        "num_utts": len(refs),
        "num_ref_words": agg.num_ref_words,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiments", nargs="+", required=True, help="name:path pairs")
    ap.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="Optional eval set overrides as name:path pairs. If omitted, uses config data.test_manifest (and config.eval_sets if present).",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    vocab = str(cfg["model"]["vocab"])
    device = torch.device(str(cfg.get("device", "cpu")))

    sets: dict[str, str] = {"test": str(cfg["data"]["test_manifest"])}
    if isinstance(cfg.get("eval_sets"), dict):
        for k, v in cfg["eval_sets"].items():
            sets[str(k)] = str(v)
    if args.sets is not None and len(args.sets) > 0:
        sets = {}
        for item in args.sets:
            if ":" not in item:
                raise ValueError(f"Invalid set spec: {item} (expected name:path)")
            name, path = item.split(":", 1)
            sets[name] = path

    results: dict = {
        "config": cfg,
        "sets": sets,
        "experiments": {},
    }

    for item in args.experiments:
        if ":" not in item:
            raise ValueError(f"Invalid experiment spec: {item} (expected name:path)")
        name, path = item.split(":", 1)
        ckpt_path = Path(path)
        model = load_model_for_ckpt(ckpt_path, cfg, vocab=vocab).to(device)
        exp_res: dict[str, object] = {"ckpt": str(ckpt_path), "per_set": {}}
        for set_name, manifest_path in sets.items():
            ds = JsonlSpeechDataset(manifest_path, target_sr=int(cfg["data"]["sample_rate"]), vocab=vocab)
            res = evaluate_one(model, ds, device=device, vocab=vocab)
            exp_res["per_set"][set_name] = res
        results["experiments"][name] = exp_res

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
