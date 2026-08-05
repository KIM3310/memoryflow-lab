from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_measurement_summary import render, validate_attention, validate_copy

ROOT = Path(__file__).resolve().parents[1]
COPY_PATH = ROOT / "evidence" / "measurements" / "apple-m4-mps-copy.json"
ATTENTION_PATH = ROOT / "evidence" / "measurements" / "apple-m4-mps-attention.json"


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_committed_measurements_validate_and_render() -> None:
    copy_payload = load_payload(COPY_PATH)
    attention_payload = load_payload(ATTENTION_PATH)
    validate_copy(copy_payload)
    validate_attention(attention_payload)
    summary = render(copy_payload, attention_payload)
    assert "37.46%" in summary
    assert "4.72%" in summary


def test_copy_derived_bandwidth_tampering_is_rejected() -> None:
    payload = copy.deepcopy(load_payload(COPY_PATH))
    payload["samples"][0]["observed_bandwidth_gbps"] = 1
    with pytest.raises(ValueError, match="observed_bandwidth"):
        validate_copy(payload)


def test_attention_modeled_byte_tampering_is_rejected() -> None:
    payload = copy.deepcopy(load_payload(ATTENTION_PATH))
    payload["samples"][0]["modeled_bytes"] += 1
    with pytest.raises(ValueError, match="attention.bytes"):
        validate_attention(payload)


def test_validation_accepts_platform_roundoff_but_rejects_material_changes() -> None:
    payload = copy.deepcopy(load_payload(ATTENTION_PATH))
    validation = payload["models"]["independent_roofline"]["validation"]
    validation["mean_absolute_percentage_error_pct"] += 1e-14
    validate_attention(payload)

    validation["mean_absolute_percentage_error_pct"] += 0.01
    with pytest.raises(ValueError, match="attention.independent_roofline"):
        validate_attention(payload)
