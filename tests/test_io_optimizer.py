from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoryflow.io import load_request, request_from_dict, write_json
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_scenario_loads() -> None:
    request = load_request(ROOT / "scenarios" / "7b-long-context-tiered.json")
    assert request.workload.parameter_count_b == 7.0
    assert request.policy.kind == "sliding_window"


def test_scenario_root_must_be_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_request(path)


def test_request_requires_all_sections() -> None:
    with pytest.raises((KeyError, TypeError)):
        request_from_dict({"workload": {}})


def test_write_json_is_stable_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    write_json(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_sweep_names_each_configuration() -> None:
    request = load_request(ROOT / "scenarios" / "7b-long-context-tiered.json")
    results = sweep_hbm_windows(request, windows=(512, 1024))
    assert [result.policy_name for result in results] == [
        "sliding_window-512",
        "sliding_window-1024",
        "near_memory-512",
        "near_memory-1024",
    ]


def test_pareto_front_excludes_infeasible_and_dominated_results() -> None:
    request = load_request(ROOT / "scenarios" / "7b-long-context-tiered.json")
    results = sweep_hbm_windows(request, windows=(128, 1024, 8192))
    frontier = pareto_front(results)
    assert frontier
    assert all(result.feasible for result in frontier)
    assert all(result.policy_name != "sliding_window-8192" for result in frontier)


def test_evidence_scenarios_answer_three_distinct_hypotheses() -> None:
    hbm = simulate(load_request(ROOT / "scenarios" / "7b-long-context-hbm-only.json"))
    tiered = simulate(load_request(ROOT / "scenarios" / "7b-long-context-tiered.json"))
    near = simulate(load_request(ROOT / "scenarios" / "7b-long-context-near-memory.json"))
    assert not hbm.feasible
    assert tiered.feasible and near.feasible
    assert near.total_remote_read_gib < tiered.total_remote_read_gib


def test_scenario_json_has_no_vendor_product_claims() -> None:
    for path in (ROOT / "scenarios").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["system"]["name"].startswith("Synthetic")
