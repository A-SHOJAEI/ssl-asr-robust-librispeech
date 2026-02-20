"""Text normalization utilities for ASR evaluation."""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    return s
