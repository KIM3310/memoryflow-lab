from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_measurement_summary import (
    render,
    validate_attention,
    validate_committed_apple_m4_pair,
    validate_copy,
)

ROOT = Path(__file__).resolve().parents[1]
COPY_PATH = ROOT / "evidence" / "measurements" / "apple-m4-mps-copy.json"
ATTENTION_PATH = ROOT / "evidence" / "measurements" / "apple-m4-mps-attention.json"


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def pair() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_payload(COPY_PATH), load_payload(ATTENTION_PATH)


def test_committed_measurements_are_aggregate_only_validate_and_render() -> None:
    copy_payload, attention_payload = pair()
    validate_committed_apple_m4_pair(copy_payload, attention_payload)
    for payload in (copy_payload, attention_payload):
        assert payload["aggregation"] == {
            "level": "summary_statistics",
            "raw_iterations_included": False,
            "reported_statistics": ["median_ms", "p95_ms"],
        }
    summary = render(copy_payload, attention_payload)
    assert "37.46%" in summary
    assert "4.72%" in summary
    assert "not raw iterations" in summary
    assert "near-memory/PIM" in summary
    assert "do not provide measurements for CUDA or other backends" in summary
    assert "Neither artifact calibrates the synthetic scenarios" in summary


def test_generic_cuda_summary_uses_the_supplied_environment_boundary() -> None:
    copy_payload, attention_payload = pair()
    for payload in (copy_payload, attention_payload):
        payload["environment"]["device_backend"] = "cuda"
        payload["environment"]["device_label"] = "Synthetic CUDA Review Device"
    summary = render(copy_payload, attention_payload)
    assert "The Synthetic CUDA Review Device CUDA artifacts" in summary
    assert "shape on CUDA" in summary
    assert "do not provide measurements for MPS or other backends" in summary
    assert "Apple M4 MPS artifacts" not in summary


def test_copy_derived_bandwidth_tampering_is_rejected() -> None:
    payload = copy.deepcopy(load_payload(COPY_PATH))
    payload["samples"][0]["observed_bandwidth_gbps"] = 1
    with pytest.raises(ValueError, match="observed_bandwidth"):
        validate_copy(payload)


def test_boolean_cannot_spoof_a_numeric_derived_field() -> None:
    payload = copy.deepcopy(load_payload(COPY_PATH))
    payload["samples"][0]["observed_bandwidth_gbps"] = True
    with pytest.raises(ValueError, match="observed_bandwidth"):
        validate_copy(payload)


def test_attention_modeled_byte_tampering_is_rejected() -> None:
    payload = copy.deepcopy(load_payload(ATTENTION_PATH))
    payload["samples"][0]["modeled_bytes"] += 1
    with pytest.raises(ValueError, match="attention.bytes"):
        validate_attention(payload)


def test_validation_accepts_roundoff_but_rejects_material_changes() -> None:
    payload = copy.deepcopy(load_payload(ATTENTION_PATH))
    validation = payload["models"]["independent_roofline"]["validation"]
    validation["mean_absolute_percentage_error_pct"] += 1e-14
    validate_attention(payload)
    validation["mean_absolute_percentage_error_pct"] += 0.01
    with pytest.raises(ValueError, match="attention.independent_roofline"):
        validate_attention(payload)


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        (
            "attention",
            lambda value: value["samples"][0].update({"median_ms": "1.0"}),
            "median_ms must be a finite number",
        ),
        (
            "attention",
            lambda value: value["samples"][0].update({"p95_ms": "1.0"}),
            "p95_ms must be a finite number",
        ),
        (
            "attention",
            lambda value: value["hardware_calibration"]["copy"]["samples"][0].update(
                {"median_ms": "1.0"}
            ),
            "attention copy median_ms must be a finite number",
        ),
        (
            "copy",
            lambda value: value["protocol"].update({"seed": "17"}),
            "copy seed must be an integer",
        ),
        (
            "attention",
            lambda value: value["protocol"].update({"timing_boundary": ""}),
            "timing_boundary must be a non-empty string",
        ),
        (
            "attention",
            lambda value: value["protocol"].update({"copy_calibration_sizes_mib": [4, 4, 64]}),
            "copy calibration sizes must be unique",
        ),
        (
            "attention",
            lambda value: value["hardware_calibration"]["copy"]["samples"].append(
                copy.deepcopy(value["hardware_calibration"]["copy"]["samples"][0])
            ),
            "copy sample sizes must be unique",
        ),
    ],
)
def test_measurement_schema_rejects_numeric_and_text_coercion(
    target: str, mutation: Any, message: str
) -> None:
    copy_payload, attention_payload = pair()
    selected = copy_payload if target == "copy" else attention_payload
    mutation(selected)
    validator = validate_copy if target == "copy" else validate_attention
    with pytest.raises(ValueError, match=message):
        validator(selected)


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("copy", lambda value: value.update({"unknown": 1}), "schema mismatch"),
        ("attention", lambda value: value["protocol"].pop("gemm_size"), "schema mismatch"),
        (
            "copy",
            lambda value: value["aggregation"].update({"raw_iterations_included": True}),
            "raw iterations",
        ),
        (
            "attention",
            lambda value: value["aggregation"].update({"level": "raw_iterations"}),
            "summary statistics",
        ),
        ("copy", lambda value: value.update({"generated_at_utc": "not-a-time"}), "ISO-8601"),
        ("attention", lambda value: value["environment"].pop("torch_version"), "schema mismatch"),
        ("copy", lambda value: value["scope"].pop("use"), "schema mismatch"),
    ],
)
def test_exact_measurement_schemas_reject_tampering(
    target: str, mutation: Any, message: str
) -> None:
    copy_payload, attention_payload = pair()
    selected = copy_payload if target == "copy" else attention_payload
    mutation(selected)
    validator = validate_copy if target == "copy" else validate_attention
    with pytest.raises(ValueError, match=message):
        validator(selected)


def test_committed_pair_requires_exact_mps_environment_and_scope_exclusions() -> None:
    copy_payload, attention_payload = pair()
    attention_payload["environment"]["torch_version"] = "different"
    with pytest.raises(ValueError, match="environments"):
        validate_committed_apple_m4_pair(copy_payload, attention_payload)

    copy_payload, attention_payload = pair()
    copy_payload["environment"]["device_backend"] = "cuda"
    attention_payload["environment"]["device_backend"] = "cuda"
    with pytest.raises(ValueError, match="Apple M4 MPS"):
        validate_committed_apple_m4_pair(copy_payload, attention_payload)

    copy_payload, attention_payload = pair()
    copy_payload["scope"]["not_measured"] = "HBM only"
    with pytest.raises(ValueError, match="exclude"):
        validate_committed_apple_m4_pair(copy_payload, attention_payload)
