PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTHONPATH := $(CURDIR)/src

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@set -euo pipefail; \
	if [ ! -x "$(PY)" ]; then \
		python3 -m venv --without-pip $(VENV); \
	fi; \
	if [ ! -x "$(PIP)" ]; then \
		mkdir -p .cache; \
		if command -v curl >/dev/null 2>&1; then \
			curl -fsSL -o .cache/get-pip.py https://bootstrap.pypa.io/get-pip.py; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO .cache/get-pip.py https://bootstrap.pypa.io/get-pip.py; \
		else \
			echo "ERROR: need curl or wget to download get-pip.py"; exit 1; \
		fi; \
		$(PY) .cache/get-pip.py; \
	fi; \
		$(PIP) install --upgrade pip setuptools wheel; \
		$(PIP) install -r requirements.txt

# By default we generate a tiny local dataset so make all is fast and offline.
data: setup
	@set -euo pipefail; \
	mkdir -p data; \
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/make_smoke_data.py --out data/smoke

train: setup data
	@set -euo pipefail; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m ssl_asr.train_asr_baseline --config $(CONFIG) --run_name smoke_baseline; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m ssl_asr.pretrain_ssl --config $(CONFIG) --run_name smoke_ssl_pretrain; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m ssl_asr.finetune_asr --config $(CONFIG) --run_name smoke_finetune --init_from runs/smoke_ssl_pretrain/checkpoints/last.pt --consistency_weight 1.0 --augment_musan true --augment_rir true; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m ssl_asr.finetune_asr --config $(CONFIG) --run_name smoke_ablation_no_consistency --init_from runs/smoke_ssl_pretrain/checkpoints/last.pt --consistency_weight 0.0 --augment_musan true --augment_rir true

eval: setup
	@set -euo pipefail; \
	mkdir -p artifacts; \
	PYTHONPATH=$(PYTHONPATH) $(PY) -m ssl_asr.evaluate_asr \
		--config $(CONFIG) \
		--experiments \
			baseline:runs/smoke_baseline/checkpoints/last.pt \
			finetune:runs/smoke_finetune/checkpoints/last.pt \
			ablation_no_consistency:runs/smoke_ablation_no_consistency/checkpoints/last.pt \
		--out artifacts/results.json

report: setup
	@set -euo pipefail; \
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/make_report.py --in artifacts/results.json --out artifacts/report.md

all: setup data train eval report

clean:
	rm -rf $(VENV) .cache artifacts runs data/smoke
