from __future__ import annotations

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, *, blank_id: int = 0) -> list[list[int]]:
    """Greedy CTC decode.

    Args:
        log_probs: (B, T, V) log-probabilities.
    Returns:
        List of token id sequences (without blanks and repeats).
    """

    ids = torch.argmax(log_probs, dim=-1)  # (B,T)
    out: list[list[int]] = []
    for b in range(ids.size(0)):
        seq: list[int] = []
        prev = None
        for t in range(ids.size(1)):
            v = int(ids[b, t].item())
            if v == blank_id:
                prev = v
                continue
            if prev is not None and v == prev:
                continue
            seq.append(v)
            prev = v
        out.append(seq)
    return out


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_id: int = 0,
) -> torch.Tensor:
    # torch.nn.CTCLoss expects (T,B,V)
    lp = log_probs.transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        lp,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
