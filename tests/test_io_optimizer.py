from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memoryflow.io import (
    MAX_SCENARIO_BYTES,
    json_object_from_bytes,
    load_request,
    request_from_dict,
    write_json,
)
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "7b-long-context-tiered.json"


def payload() -> dict[str, Any]:
    value = json.loads(SCENARIO.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_all_bundled_scenarios_load_with_synthetic_provenance() -> None:
    for path in sorted((ROOT / "scenarios").glob("*.json")):
        request = load_request(path)
        assert request.workload.parameter_count_b == 7.0
        assert request.system.name.startswith("Synthetic")
        assert request.provenance.hardware_profile == "synthetic"
        assert request.provenance.measurement_scope == "none"


def test_scenario_root_must_be_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_request(path)
    with pytest.raises(ValueError, match="root"):
        request_from_dict([])


@pytest.mark.parametrize("section", ["workload", "system", "policy", "provenance"])
def test_exact_nested_schema_rejects_missing_and_unknown(section: str) -> None:
    missing = payload()
    removed = next(iter(missing[section]))
    missing[section].pop(removed)
    with pytest.raises(ValueError, match=f"missing {section} fields"):
        request_from_dict(missing)

    unknown = payload()
    unknown[section]["unexpected"] = 1
    with pytest.raises(ValueError, match=f"unknown {section} fields"):
        request_from_dict(unknown)


def test_exact_root_schema_and_version_are_required() -> None:
    missing = payload()
    missing.pop("policy")
    with pytest.raises(ValueError, match="missing scenario fields"):
        request_from_dict(missing)
    unknown = payload()
    unknown["extra"] = {}
    with pytest.raises(ValueError, match="unknown scenario fields"):
        request_from_dict(unknown)
    version = payload()
    version["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="unsupported scenario schema_version"):
        request_from_dict(version)


def test_duplicate_keys_and_non_standard_numbers_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"2.0","schema_version":"2.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_request(duplicate)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(SCENARIO.read_text().replace("7.0", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON number"):
        load_request(invalid)
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_request(invalid_utf8)


def test_json_parser_converts_excessive_nesting_to_a_contract_error() -> None:
    nested = b'{"x":' * 2_000 + b"0" + b"}" * 2_000
    with pytest.raises(ValueError, match="nesting exceeds"):
        json_object_from_bytes(nested)


def test_json_parser_rejects_unpaired_surrogates_but_accepts_valid_pairs() -> None:
    for raw in (
        b'{"value":"\ud800"}',
        b'{"\udfff":1}',
        b'{"\ud800":1,"\ud800":2}',
    ):
        with pytest.raises(ValueError, match="valid Unicode scalar values"):
            json_object_from_bytes(raw)
    parsed = json_object_from_bytes(b'{"emoji":"\ud83d\ude00"}')
    assert parsed == {"emoji": "😀"}
    with pytest.raises(ValueError, match="numbers must be finite"):
        json_object_from_bytes(b'{"overflow":1e999}')


def test_scenario_size_limit_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * (MAX_SCENARIO_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_request(path)


def test_write_json_is_stable_finite_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    write_json(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})


def test_sweep_is_page_aligned_and_deterministic() -> None:
    request = load_request(SCENARIO)
    results = sweep_hbm_windows(request, windows=(512, 1024))
    assert [result.policy_name for result in results] == [
        "sliding_window-512",
        "sliding_window-1024",
        "near_memory-512",
        "near_memory-1024",
    ]
    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in sweep_hbm_windows(request, windows=(512, 1024))
    ]
    for invalid in ((), (512, 512), (511,), (True,)):
        with pytest.raises(ValueError, match="windows|window"):
            sweep_hbm_windows(request, windows=invalid)  # type: ignore[arg-type]


def test_pareto_front_excludes_infeasible_and_dominated_results() -> None:
    request = load_request(SCENARIO)
    results = sweep_hbm_windows(request, windows=(128, 1024, 8192))
    frontier = pareto_front(results)
    assert frontier
    assert all(result.feasible for result in frontier)
    assert all(result.policy_name != "sliding_window-8192" for result in frontier)
    assert frontier == pareto_front(list(reversed(results)))


def test_evidence_scenarios_answer_distinct_hypotheses() -> None:
    hbm = simulate(load_request(ROOT / "scenarios" / "7b-long-context-hbm-only.json"))
    tiered = simulate(load_request(ROOT / "scenarios" / "7b-long-context-tiered.json"))
    near = simulate(load_request(ROOT / "scenarios" / "7b-long-context-near-memory.json"))
    stress = simulate(load_request(ROOT / "scenarios" / "7b-long-context-near-memory-stress.json"))
    assert not hbm.feasible
    assert tiered.feasible and near.feasible and stress.feasible
    assert near.total_remote_memory_read_gib == pytest.approx(tiered.total_remote_memory_read_gib)
    assert near.total_interconnect_read_gib < tiered.total_interconnect_read_gib
    assert (
        near.mean_decode_latency_ms < tiered.mean_decode_latency_ms < stress.mean_decode_latency_ms
    )
