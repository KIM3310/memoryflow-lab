PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test lint typecheck verify benchmark run docker-build docker-verify docker-up docker-down verify-all

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

verify-all: verify docker-verify

benchmark:
	$(BIN)/python -m scripts.build_evidence
	git diff --exit-code -- site/results.json evidence/benchmark-summary.md

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
