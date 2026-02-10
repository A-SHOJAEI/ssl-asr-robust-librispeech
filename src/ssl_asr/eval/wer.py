from __future__ import annotations

from dataclasses import dataclass

import jiwer


@dataclass(frozen=True)
class WerResult:
    wer: float
    num_ref_words: int


def compute_wer(refs: list[str], hyps: list[str]) -> WerResult:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")

    # jiwer computes WER as (S+D+I)/N.
    tr = jiwer.process_words(refs, hyps)
    n = int(tr.hits + tr.substitutions + tr.deletions)
    wer = float("nan") if n == 0 else float((tr.substitutions + tr.deletions + tr.insertions) / n)
    return WerResult(wer=wer, num_ref_words=n)
