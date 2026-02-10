from __future__ import annotations

import argparse
import random
from pathlib import Path

from ssl_asr.data.augment import list_audio_files
from ssl_asr.utils.io import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True, help="dir containing clean manifests")
    ap.add_argument("--musan", required=True, help="path to extracted MUSAN root")
    ap.add_argument("--rirs", required=True, help="path to extracted RIRS_NOISES root")
    ap.add_argument("--snrs", required=True, help="comma-separated SNR list, e.g. 0,5,10,20")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--splits", default="test-clean,test-other", help="comma-separated eval splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(int(args.seed))

    manifests_dir = Path(args.manifests)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    musan_files = list_audio_files(Path(args.musan))
    rir_files = list_audio_files(Path(args.rirs))
    if not musan_files:
        raise RuntimeError(f"No audio files found under MUSAN dir: {args.musan}")
    if not rir_files:
        raise RuntimeError(f"No audio files found under RIRS dir: {args.rirs}")

    snrs = [float(x.strip()) for x in args.snrs.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        clean_path = manifests_dir / f"{split}.jsonl"
        rows = read_jsonl(clean_path)

        # Noise-only variants at fixed SNR.
        for snr in snrs:
            out_rows = []
            for r in rows:
                noise = rng.choice(musan_files)
                out_rows.append(
                    {
                        **r,
                        "aug": {
                            "noise_path": str(noise.resolve()),
                            "snr_db": snr,
                        },
                    }
                )
            out_path = out_dir / f"{split}_musan_snr{int(snr)}.jsonl"
            write_jsonl(out_path, out_rows)
            print(f"Wrote {out_path} ({len(out_rows)} rows)")

        # RIR-only variants.
        out_rows = []
        for r in rows:
            rir = rng.choice(rir_files)
            out_rows.append({**r, "aug": {"rir_path": str(rir.resolve())}})
        out_path = out_dir / f"{split}_rir.jsonl"
        write_jsonl(out_path, out_rows)
        print(f"Wrote {out_path} ({len(out_rows)} rows)")

        # Combined RIR + noise.
        for snr in snrs:
            out_rows = []
            for r in rows:
                rir = rng.choice(rir_files)
                noise = rng.choice(musan_files)
                out_rows.append(
                    {
                        **r,
                        "aug": {
                            "rir_path": str(rir.resolve()),
                            "noise_path": str(noise.resolve()),
                            "snr_db": snr,
                        },
                    }
                )
            out_path = out_dir / f"{split}_rir_musan_snr{int(snr)}.jsonl"
            write_jsonl(out_path, out_rows)
            print(f"Wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
