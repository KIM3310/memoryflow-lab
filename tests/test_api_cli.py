from __future__ import annotations

import json
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from memoryflow import __version__
from memoryflow.api import app, resolve_site_directory
from memoryflow.cli import main

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "7b-long-context-tiered.json"


def test_health_endpoint_declares_model_boundary() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "analytical-first-order"}


def test_openapi_reports_package_version() -> None:
    response = TestClient(app).get("/openapi.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert __version__ == project["project"]["version"]
    assert response.json()["info"]["version"] == __version__


def test_simulation_endpoint() -> None:
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    response = TestClient(app).post("/v1/simulations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["feasible"] is True
    assert len(body["steps"]) == 64


def test_simulation_endpoint_rejects_invalid_payload() -> None:
    response = TestClient(app).post("/v1/simulations", json={"workload": {}})
    assert response.status_code == 422


def test_simulation_endpoint_rejects_non_finite_numbers() -> None:
    payload = SCENARIO.read_text(encoding="utf-8").replace(
        '"parameter_count_b": 7.0', '"parameter_count_b": NaN'
    )
    response = TestClient(app).post(
        "/v1/simulations",
        content=payload,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "positive finite number" in response.json()["detail"]


def test_dashboard_is_served() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "MemoryFlow Lab" in response.text


def test_dashboard_directory_can_be_configured(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORYFLOW_SITE_DIR", str(tmp_path))
    assert resolve_site_directory() == tmp_path.resolve()


def test_cli_simulate_writes_summary(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    assert main(["simulate", str(SCENARIO), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["feasible"] is True
    assert "steps" not in payload


def test_cli_simulate_can_include_steps(tmp_path: Path) -> None:
    output = tmp_path / "detail.json"
    assert main(["simulate", str(SCENARIO), "--steps", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 64


def test_cli_optimize_writes_frontier(tmp_path: Path) -> None:
    output = tmp_path / "frontier.json"
    assert main(["optimize", str(SCENARIO), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all"]
    assert payload["pareto_front"]
