from __future__ import annotations

from dataclasses import replace

import pytest

from memoryflow.domain import PlacementPolicy, SimulationRequest
from memoryflow.simulator import ASSUMPTIONS, simulate
from tests.test_domain import sample_system, sample_workload


def request(kind: str = "sliding_window", window: int = 1024) -> SimulationRequest:
    workload = sample_workload(context_tokens=8192, generated_tokens=16, concurrent_sequences=16)
    policy = PlacementPolicy(
        name=kind,
        kind=kind,  # type: ignore[arg-type]
        hbm_window_tokens=window,
        transfer_overlap_ratio=0.35,
        near_memory_reduction_ratio=0.90,
    )
    return SimulationRequest(workload, sample_system(), policy)


def test_hbm_only_rejects_capacity_overflow() -> None:
    result = simulate(request("hbm_only", 8192))
    assert not result.feasible
    assert result.rejection_reason == "weights plus resident KV cache exceed HBM capacity"
    assert result.bottleneck == "capacity"


def test_tiered_memory_restores_feasibility() -> None:
    result = simulate(request())
    assert result.feasible
    assert result.peak_remote_kv_gib > 0
    assert len(result.steps) == 16


def test_near_memory_reduces_remote_traffic_and_latency() -> None:
    tiered = simulate(request("sliding_window"))
    near = simulate(request("near_memory"))
    assert near.total_remote_read_gib == pytest.approx(tiered.total_remote_read_gib * 0.1)
    assert near.mean_decode_latency_ms < tiered.mean_decode_latency_ms
    assert near.throughput_tokens_s > tiered.throughput_tokens_s


def test_slow_near_memory_compute_can_erase_the_transfer_gain() -> None:
    baseline = request("near_memory")
    slow_system = replace(baseline.system, near_memory_tops=0.1)
    slow_near = simulate(replace(baseline, system=slow_system))
    tiered = simulate(request("sliding_window"))
    assert slow_near.mean_decode_latency_ms > tiered.mean_decode_latency_ms
    assert slow_near.bottleneck == "near_memory_compute"


def test_more_remote_bandwidth_lowers_tiered_latency() -> None:
    baseline_request = request()
    faster = replace(
        baseline_request,
        system=replace(baseline_request.system, remote_bandwidth_gbps=128),
    )
    assert (
        simulate(faster).mean_decode_latency_ms < simulate(baseline_request).mean_decode_latency_ms
    )


def test_more_overlap_lowers_exposed_remote_latency() -> None:
    baseline_request = request()
    overlapped = replace(
        baseline_request,
        policy=replace(baseline_request.policy, transfer_overlap_ratio=0.9),
    )
    assert (
        simulate(overlapped).mean_decode_latency_ms
        < simulate(baseline_request).mean_decode_latency_ms
    )


def test_larger_window_reduces_remote_read() -> None:
    small = simulate(request(window=256))
    large = simulate(request(window=2048))
    assert large.total_remote_read_gib < small.total_remote_read_gib
    assert large.peak_hbm_kv_gib > small.peak_hbm_kv_gib


def test_too_large_window_can_overflow_hbm() -> None:
    assert not simulate(request(window=8192)).feasible


def test_model_weight_overflow_is_reported_before_kv() -> None:
    base = request()
    oversized = replace(base, workload=replace(base.workload, parameter_count_b=30))
    result = simulate(oversized)
    assert not result.feasible
    assert result.rejection_reason == "model weights exceed HBM capacity"


def test_remote_capacity_overflow_is_reported() -> None:
    base = request()
    constrained = replace(base, system=replace(base.system, remote_capacity_gib=1))
    result = simulate(constrained)
    assert not result.feasible
    assert result.rejection_reason == "cold KV cache exceeds remote-memory capacity"


def test_result_summary_is_consistent() -> None:
    result = simulate(request("near_memory"))
    summary = result.to_dict(include_steps=False)
    assert result.p95_step_latency_ms >= result.mean_decode_latency_ms
    assert result.estimated_energy_j > 0
    assert result.total_hbm_read_gib > 0
    assert result.assumptions == ASSUMPTIONS
    assert summary.get("steps") is None
    assert summary["total_remote_read_gib"] == round(result.total_remote_read_gib, 10)
    assert len(result.to_dict(include_steps=True)["steps"]) == 16
    assert result.steps[0].near_memory_compute_ms > 0


def test_simulation_is_deterministic() -> None:
    first = simulate(request()).to_dict()
    second = simulate(request()).to_dict()
    assert first == second
