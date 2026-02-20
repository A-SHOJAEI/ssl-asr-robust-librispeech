"""JSONL manifest loading and speech dataset for ASR training/eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class JsonlSpeechDataset(Dataset):
    """Load utterances from a JSONL manifest.

    Each manifest row must have at least 'audio_path' and 'text'.
    Optional: 'sample_rate', 'duration_s', 'aug' (for robust eval).
    """

    def __init__(self, manifest_path: str | Path, target_sr: int = 16000, vocab: str = "") -> None:
        self.items = _read_jsonl(manifest_path)
        self.target_sr = target_sr
        self.vocab = vocab
        self._ctoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 is CTC blank

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        audio, sr = sf.read(item["audio_path"], dtype="float32")
        wav = torch.from_numpy(audio).float()

        text = str(item.get("text", ""))
        tokens = [self._ctoi[c] for c in text if c in self._ctoi]
        token_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            "wav": wav,
            "wav_len": wav.shape[0],
            "tokens": token_ids,
            "tok_len": token_ids.shape[0],
            "text": text,
            "aug": item.get("aug"),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate list of dataset items into a padded batch."""
    wavs = [b["wav"] for b in batch]
    tokens = [b["tokens"] for b in batch]

    wav_lens = torch.tensor([b["wav_len"] for b in batch], dtype=torch.long)
    tok_lens = torch.tensor([b["tok_len"] for b in batch], dtype=torch.long)

    wav_padded = pad_sequence(wavs, batch_first=True, padding_value=0.0)
    tok_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    return {
        "wav": wav_padded,
        "wav_lens": wav_lens,
        "tokens": tok_padded,
        "tok_lens": tok_lens,
        "text": [b["text"] for b in batch],
        "aug": [b.get("aug") for b in batch],
    }
