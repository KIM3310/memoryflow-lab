from __future__ import annotations

import json
import threading
import time
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from memoryflow import __version__
from memoryflow.api import app, resolve_site_directory
from memoryflow.cli import main
from memoryflow.domain import MIN_EFFECTIVE_RATE

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "7b-long-context-tiered.json"


def test_health_openapi_and_dashboard() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "model": "analytical-first-order"}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    schema = openapi.json()
    assert __version__ == project["project"]["version"] == schema["info"]["version"]
    for route in ("/v1/simulations", "/v1/analyses"):
        request_body = schema["paths"][route]["post"]["requestBody"]
        assert request_body["required"] is True
        assert "application/json" in request_body["content"]
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "MemoryFlow Lab" in dashboard.text


def test_simulation_and_analysis_endpoints() -> None:
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    client = TestClient(app)
    simulation = client.post("/v1/simulations", json=payload)
    assert simulation.status_code == 200
    body = simulation.json()
    assert body["schema_version"] == "2.0"
    assert body["feasible"] is True
    assert len(body["steps"]) == 64

    analysis = client.post("/v1/analyses", json=payload)
    assert analysis.status_code == 200
    assert analysis.json()["schema_version"] == "1.0"
    assert analysis.json()["counterexample"]["counterexample_winner"] == "sliding_window"


def test_analysis_api_returns_explicit_unreachable_counterexample_at_rate_floor() -> None:
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    efficiency = payload["system"]["remote_bandwidth_efficiency"]
    payload["system"]["remote_bandwidth_gbps"] = MIN_EFFECTIVE_RATE * (1 + 1e-12) / efficiency
    response = TestClient(app).post("/v1/analyses", json=payload)
    assert response.status_code == 200
    counterexample = response.json()["counterexample"]
    assert counterexample["status"] == "not_reachable_within_bounds"
    assert counterexample["counterexample_winner"] is None


def test_cpu_bound_analysis_runs_off_the_event_loop(monkeypatch: MonkeyPatch) -> None:
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    entered = threading.Event()
    release = threading.Event()
    result: dict[str, object] = {}

    class SlowReport:
        def to_dict(self) -> dict[str, str]:
            return {"schema_version": "test"}

    def slow_analysis(_request: object) -> SlowReport:
        entered.set()
        release.wait(timeout=2)
        return SlowReport()

    monkeypatch.setattr("memoryflow.api.analyze_design_space", slow_analysis)
    with TestClient(app) as client:
        worker = threading.Thread(
            target=lambda: result.update(response=client.post("/v1/analyses", json=payload))
        )
        worker.start()
        assert entered.wait(timeout=1)
        started = time.monotonic()
        health = client.get("/health")
        elapsed = time.monotonic() - started
        release.set()
        worker.join(timeout=2)
    assert health.status_code == 200
    assert elapsed < 0.5
    assert not worker.is_alive()
    response = result["response"]
    assert response.status_code == 200  # type: ignore[attr-defined]


def test_endpoints_reject_invalid_payloads_and_non_finite_numbers() -> None:
    client = TestClient(app)
    for route in ("/v1/simulations", "/v1/analyses"):
        response = client.post(route, json={"workload": {}})
        assert response.status_code == 422
        assert "missing scenario fields" in response.json()["detail"]

    raw = SCENARIO.read_text(encoding="utf-8").replace(
        '"parameter_count_b": 7.0', '"parameter_count_b": NaN'
    )
    response = client.post(
        "/v1/simulations",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "non-standard JSON number" in response.json()["detail"]

    subnormal = json.loads(SCENARIO.read_text(encoding="utf-8"))
    subnormal["system"]["hbm_bandwidth_gbps"] = 5e-324
    response = client.post("/v1/simulations", json=subnormal)
    assert response.status_code == 422
    assert "effective rates must be at least" in response.json()["detail"]
    assert "null" not in response.text


def test_api_rejects_valid_json_with_unsupported_media_type() -> None:
    response = TestClient(app).post(
        "/v1/simulations",
        content=SCENARIO.read_bytes(),
        headers={"content-type": "text/plain; charset=utf-8"},
    )
    assert response.status_code == 415
    assert response.json()["detail"] == "Content-Type must be application/json"


def test_api_rejects_oversized_body_before_json_parsing() -> None:
    oversized = b"{" + b" " * 1_048_576
    response = TestClient(app).post(
        "/v1/simulations",
        content=oversized,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "exceeds 1,048,576 bytes" in response.json()["detail"]


def test_api_rejects_unpaired_unicode_surrogates_without_response_encoding_failure() -> None:
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    payload["workload"]["name"] = "\ud800"
    raw = json.dumps(payload).encode("utf-8")
    for route in ("/v1/simulations", "/v1/analyses"):
        response = TestClient(app, raise_server_exceptions=False).post(
            route,
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert "valid Unicode scalar values" in response.json()["detail"]


def test_api_rejects_excessive_json_nesting_without_internal_error() -> None:
    nested = '{"x":' * 2_000 + "0" + "}" * 2_000
    for route in ("/v1/simulations", "/v1/analyses"):
        response = TestClient(app, raise_server_exceptions=False).post(
            route,
            content=nested,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "JSON nesting exceeds the parser limit"


def test_api_rejects_duplicate_json_keys_before_schema_construction() -> None:
    raw = SCENARIO.read_text(encoding="utf-8").replace(
        '"schema_version": "2.0",',
        '"schema_version": "2.0", "schema_version": "2.0",',
        1,
    )
    response = TestClient(app).post(
        "/v1/simulations",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "duplicate JSON key" in response.json()["detail"]


def test_dashboard_directory_can_be_configured(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORYFLOW_SITE_DIR", str(tmp_path))
    assert resolve_site_directory() == tmp_path.resolve()
    missing = tmp_path / "missing"
    monkeypatch.setenv("MEMORYFLOW_SITE_DIR", str(missing))
    try:
        resolve_site_directory()
    except RuntimeError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing site directory should fail")


def test_cli_simulate_optimize_and_analyze_write_versioned_outputs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    detail = tmp_path / "detail.json"
    frontier = tmp_path / "frontier.json"
    analysis = tmp_path / "analysis.json"
    overflow_analysis = tmp_path / "overflow-analysis.json"
    assert main(["simulate", str(SCENARIO), "--output", str(summary)]) == 0
    assert main(["simulate", str(SCENARIO), "--steps", "--output", str(detail)]) == 0
    assert (
        main(["optimize", str(SCENARIO), "--windows", "512,1024", "--output", str(frontier)]) == 0
    )
    assert (
        main(
            [
                "analyze",
                str(SCENARIO),
                "--link-bandwidths",
                "64",
                "--near-memory-tops",
                "0.1,12",
                "--sensitivity-multipliers",
                "0.5,1,1.5",
                "--output",
                str(analysis),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "analyze",
                str(SCENARIO),
                "--link-bandwidths",
                "64",
                "--near-memory-tops",
                "12",
                "--sensitivity-multipliers",
                "1e308",
                "--output",
                str(overflow_analysis),
            ]
        )
        == 0
    )
    overflow_payload = json.loads(overflow_analysis.read_text())
    assert any(point["input_value"] is None for point in overflow_payload["sensitivity"])
    assert "steps" not in json.loads(summary.read_text())
    assert len(json.loads(detail.read_text())["steps"]) == 64
    assert json.loads(frontier.read_text())["schema_version"] == "1.0"
    assert len(json.loads(frontier.read_text())["all"]) == 4
    assert json.loads(analysis.read_text())["schema_version"] == "1.0"


def test_cli_stdout_is_valid_json(capsys: object) -> None:
    # `capsys` remains untyped to keep this test independent of pytest's fixture types.
    assert main(["simulate", str(SCENARIO)]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(output)["schema_version"] == "2.0"
