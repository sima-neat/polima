SHELL := /bin/bash
VENV_PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3)
LEROBOT_SOURCE ?=

.PHONY: help
help:
	@echo "PoLiMa - policy framework for SiMa Modalix"
	@echo ""
	@echo "  make venv LEROBOT_SOURCE=/path/to/lerobot   create/update the dedicated .venv"
	@echo "  make install              editable install into the current env"
	@echo "  make test                 unit tests"
	@echo "  make doctor               environment diagnosis"
	@echo "  make lint                 ruff"

.PHONY: install
install:
	pip install -e ".[dev]"

.PHONY: venv
venv:
	@test -n "$(LEROBOT_SOURCE)" || { echo "set LEROBOT_SOURCE=/path/to/your/lerobot checkout" >&2; exit 2; }
	@test -x .venv/bin/python || $(VENV_PYTHON) -m venv .venv
	@.venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || { \
		echo ".venv uses Python $$(.venv/bin/python -c 'import platform; print(platform.python_version())'); compiling needs Python 3.12." >&2; \
		echo "Remove .venv and rerun with Python 3.12 installed (or VENV_PYTHON=/path/to/python3.12)." >&2; exit 2; }
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
		'torch==2.3.1+cpu' 'torchvision==0.18.1+cpu'
	.venv/bin/python -m pip install -c constraints-host.txt -e ".[host,dev]"
	.venv/bin/python scripts/install_lerobot_compat.py \
		"$(LEROBOT_SOURCE)" constraints-host.txt
	@echo "PoLiMa CPU-only venv ready: $$(.venv/bin/python --version) at $$(pwd)/.venv"

.PHONY: test
test:
	@test -x .venv/bin/pytest || { echo "run 'make venv' first" >&2; exit 2; }
	.venv/bin/pytest -q tests/unit

.PHONY: doctor
doctor:
	.venv/bin/polima doctor

.PHONY: lint
lint:
	@test -x .venv/bin/ruff || { echo "run 'make venv' first" >&2; exit 2; }
	.venv/bin/ruff check src tests
