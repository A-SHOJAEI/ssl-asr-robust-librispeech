from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    inp = Path(args.inp)
    out = Path(args.out)

    data = json.loads(inp.read_text(encoding="utf-8"))

    exp = data.get("experiments", {})
    sets = data.get("sets", {"test": data.get("test_manifest", "")})

    lines: list[str] = []
    lines.append("# ASR Robustness Report\n")
    lines.append("This report is generated from `artifacts/results.json`.\n")

    lines.append("## Summary\n")
    for set_name in sets.keys():
        lines.append(f"### Set: `{set_name}`\n")
        lines.append("| Experiment | WER | 95% CI | #utts | #ref words |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for name, r in exp.items():
            per_set = (r.get("per_set") or {}).get(set_name, {})
            wer = float(per_set.get("wer", float("nan")))
            ci = per_set.get("wer_ci", {})
            lines.append(
                f"| {name} | {wer:.4f} | [{float(ci.get('lo', float('nan'))):.4f}, {float(ci.get('hi', float('nan'))):.4f}] | {per_set.get('num_utts')} | {per_set.get('num_ref_words')} |\n"
            )
        lines.append("\n")

    lines.append("\n## Notes\n")
    lines.append("- Greedy CTC decoding only (LM decoding is optional and not enabled by default).\n")
    lines.append("- The default `configs/smoke.yaml` uses a tiny synthetic dataset; WER values are not meaningful for research conclusions.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
