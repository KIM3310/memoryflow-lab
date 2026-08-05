PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install install-measure test lint typecheck verify benchmark measurement-check measure measure-copy measure-attention run docker-build docker-verify docker-up docker-down verify-all

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
	$(BIN)/mypy src

verify: lint typecheck test benchmark measurement-check

verify-all: verify docker-verify

benchmark:
	$(BIN)/python -m scripts.build_evidence
	git diff --exit-code -- site/results.json evidence/benchmark-summary.md

measurement-check:
	$(BIN)/python -m scripts.build_measurement_summary --check

measure: measure-copy measure-attention

measure-copy:
	$(BIN)/python -m scripts.measure_torch --device auto

measure-attention:
	$(BIN)/python -m scripts.measure_attention --device auto

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
