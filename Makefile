PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install install-measure test lint typecheck verify benchmark benchmark-determinism measurement-check site-check package-check measure measure-copy measure-attention profile-cuda run docker-build docker-verify docker-up docker-down verify-all

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

install-measure: install
	$(BIN)/pip install -e ".[measure]"

test:
	$(BIN)/pytest --cov=memoryflow --cov-report=term-missing --cov-fail-under=90

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

typecheck:
	$(BIN)/mypy src scripts

verify: lint typecheck test benchmark benchmark-determinism measurement-check site-check package-check

verify-all: verify docker-verify

benchmark:
	$(BIN)/python -m scripts.build_evidence --check

benchmark-determinism:
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
		cp site/results.json "$$tmp/results.json"; \
		cp evidence/benchmark-summary.md "$$tmp/benchmark-summary.md"; \
		$(BIN)/python -m scripts.build_evidence >/dev/null; \
		cmp -s site/results.json "$$tmp/results.json"; \
		cmp -s evidence/benchmark-summary.md "$$tmp/benchmark-summary.md"

measurement-check:
	$(BIN)/python -m scripts.build_measurement_summary --check

site-check:
	$(BIN)/python -m scripts.validate_site

package-check:
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
		$(BIN)/python -m build --outdir "$$tmp" >/dev/null; \
		$(BIN)/python -m scripts.validate_distribution "$$tmp"

measure: measure-copy measure-attention

measure-copy:
	$(BIN)/python -m scripts.measure_torch --device auto

measure-attention:
	$(BIN)/python -m scripts.measure_attention --device auto

profile-cuda:
	$(BIN)/python -m scripts.profile_attention

run:
	$(BIN)/uvicorn memoryflow.api:app --reload

docker-build:
	docker build --tag memoryflow-lab:local .

docker-verify:
	bash scripts/smoke_container.sh

docker-up:
	docker compose up --build --detach --wait

docker-down:
	docker compose down --remove-orphans
