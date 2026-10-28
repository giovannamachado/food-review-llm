PY ?= python
.PHONY: install data baselines test lint lora-dry
install: ; $(PY) -m pip install -e ".[dev]"
data: ; $(PY) scripts/make_data.py
baselines: ; $(PY) scripts/run_baselines.py
test: ; $(PY) -m pytest -q
lint: ; ruff check src scripts tests
lora-dry: ; $(PY) scripts/train_lora.py --task sentiment --dry-run
