PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test lint typecheck verify benchmark run

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest --cov=memoryflow --cov-report=term-missing --cov-fail-under=90

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

typecheck:
	$(BIN)/mypy src

verify: lint typecheck test benchmark

benchmark:
	$(BIN)/python -m scripts.build_evidence
	git diff --exit-code -- site/results.json evidence/benchmark-summary.md

run:
	$(BIN)/uvicorn memoryflow.api:app --reload

