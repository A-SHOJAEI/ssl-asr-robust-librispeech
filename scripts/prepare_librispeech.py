from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from ssl_asr.data.text import normalize_text
from ssl_asr.utils.io import write_jsonl


def parse_transcripts(chapter_dir: Path) -> dict[str, str]:
    # Each chapter dir has a <speaker>-<chapter>.trans.txt
    trans_files = list(chapter_dir.glob("*.trans.txt"))
    out: dict[str, str] = {}
    for tf in trans_files:
        for line in tf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, *words = line.split()
            out[utt_id] = normalize_text(" ".join(words))
    return out


def subset_root(raw_dir: Path) -> Path:
    # OpenSLR LibriSpeech tarballs extract to LibriSpeech/
    # Accept either raw_dir/LibriSpeech or raw_dir directly.
    if (raw_dir / "LibriSpeech").is_dir():
        return raw_dir / "LibriSpeech"
    return raw_dir


def build_manifest(ls_root: Path, subset: str) -> list[dict]:
    subset_dir = ls_root / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset dir not found: {subset_dir}")

    rows: list[dict] = []
    for speaker_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
        for chapter_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
            trans = parse_transcripts(chapter_dir)
            for audio in sorted(chapter_dir.glob("*.flac")):
                utt_id = audio.stem
                text = trans.get(utt_id)
                if text is None:
                    continue
                info = sf.info(str(audio))
                rows.append(
                    {
                        "audio_path": str(audio.resolve()),
                        "text": text,
                        "sample_rate": int(info.samplerate),
                        "duration_s": float(info.frames / info.samplerate),
                        "utt_id": utt_id,
                        "subset": subset,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="data/raw where LibriSpeech is extracted")
    ap.add_argument("--out", required=True, help="output dir for manifests")
    ap.add_argument("--subsets", required=True, help="comma-separated subsets, e.g. train-clean-100,dev-clean,test-clean")
    args = ap.parse_args(argv)

    inp = subset_root(Path(args.inp))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        rows = build_manifest(inp, subset)
        mpath = out / f"{subset}.jsonl"
        write_jsonl(mpath, rows)
        print(f"Wrote {mpath} ({len(rows)} utterances)")


if __name__ == "__main__":
    main()
