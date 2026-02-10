Overwrote `README.md` with a project-specific, code-and-artifacts-grounded writeup including:

- Problem statement and exact dataset provenance (OpenSLR SLR12/17/28 + optional SLR11), tied to `scripts/download_openslr.py`, `scripts/prepare_librispeech.py`, `scripts/make_noisy_eval.py`
- Methodology as implemented (baseline `ConformerCTC`, simplified wav2vec2-style SSL, wav2vec2-CTC fine-tuning with MUSAN/RIR augment + symmetric-KL consistency)
- Baselines/ablations that exist as real flags/entrypoints
- Exact results reproduced from `artifacts/results.json` / `artifacts/report.md` (including checkpoint paths, WER, CI, #utts/#words) and metric computation details
- Repro instructions for smoke and real-data runs (including the required `--extract` for downloads)
- Limitations and concrete next research steps aligned to current code/configs

File: `README.md`