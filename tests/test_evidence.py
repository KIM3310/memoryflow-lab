from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_evidence import SCENARIOS, build, check, validate_payload
from scripts.validate_site import validate_site

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "site" / "results.json"


def load_results() -> dict[str, Any]:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_committed_evidence_validates_and_build_is_deterministic() -> None:
    before = RESULTS.read_bytes()
    payload = build()
    validate_payload(payload)
    check()
    assert RESULTS.read_bytes() == before
    assert json.loads(json.dumps(payload)) == load_results()
    assert payload["schema_version"] == "2.0"
    assert len(payload["results"]) == 4
    assert payload["analysis"]["counterexample"]["counterexample_winner"] == "sliding_window"


def test_evidence_manifest_has_exact_source_hashes_and_boundaries() -> None:
    payload = load_results()
    scenarios = payload["inputs"]["scenarios"]
    measurements = payload["inputs"]["measurements"]
    assert [record["path"] for record in scenarios] == [
        path.relative_to(ROOT).as_posix() for path in SCENARIOS
    ]
    assert all(record["hardware_profile"] == "synthetic" for record in scenarios)
    assert all(record["measurement_scope"] == "none" for record in scenarios)
    assert all(record["device_backend"] == "mps" for record in measurements)
    assert all(record["raw_iterations_included"] is False for record in measurements)
    assert all(result["input_hardware_profile"] == "synthetic" for result in payload["results"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"extra": 1}), "schema mismatch"),
        (lambda value: value["generator"].update({"package_version": "old"}), "stale"),
        (
            lambda value: value["generator"]["model_sources"][0].update({"extra": "x"}),
            "schema mismatch",
        ),
        (
            lambda value: value["generator"]["model_sources"][0].update({"sha256": "0" * 64}),
            "model-source provenance",
        ),
        (lambda value: value["inputs"]["scenarios"][0].update({"sha256": "0" * 64}), "provenance"),
        (
            lambda value: value["inputs"]["scenarios"][0].update({"hardware_profile": "measured"}),
            "synthetic",
        ),
        (
            lambda value: value["inputs"]["measurements"][0].update(
                {"raw_iterations_included": True}
            ),
            "boundary",
        ),
        (lambda value: value["results"][0].update({"schema_version": "1.0"}), "result"),
        (lambda value: value["results"][0].update({"unknown": 1}), "schema mismatch"),
        (lambda value: value["analysis"].update({"schema_version": "old"}), "analysis"),
        (lambda value: value["analysis"].update({"unknown": 1}), "schema mismatch"),
    ],
)
def test_evidence_schema_and_provenance_tampering_is_rejected(mutation: Any, message: str) -> None:
    payload = copy.deepcopy(load_results())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_payload(payload)


def test_embedded_measurement_must_equal_source_artifact() -> None:
    payload = copy.deepcopy(load_results())
    payload["measurement"]["samples"][0]["median_ms"] += 0.01
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_static_site_has_required_ids_assets_fields_and_valid_payload() -> None:
    validate_site()
