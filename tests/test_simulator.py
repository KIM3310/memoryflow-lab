from __future__ import annotations

from dataclasses import replace

import pytest

from memoryflow.domain import GIB, PlacementPolicy, SimulationRequest
from memoryflow.simulator import ASSUMPTIONS, simulate
from tests.test_domain import sample_system, sample_workload


def request(
    kind: str = "sliding_window",
    window: int = 1024,
    *,
    page_tokens: int = 16,
    context_tokens: int = 8193,
    generated_tokens: int = 3,
) -> SimulationRequest:
    workload = sample_workload(
        context_tokens=context_tokens,
        generated_tokens=generated_tokens,
        concurrent_sequences=16,
    )
    policy = PlacementPolicy(
        name=kind,
        kind=kind,  # type: ignore[arg-type]
        hbm_window_tokens=window,
        kv_page_tokens=page_tokens,
        transfer_overlap_ratio=0.35,
        near_memory_accumulator_bits=32,
    )
    return SimulationRequest(workload, sample_system(), policy)


def test_hbm_only_rejects_post_write_page_capacity_overflow() -> None:
    result = simulate(request("hbm_only", 8192, context_tokens=8192, generated_tokens=16))
    assert not result.feasible
    assert result.rejection_reason == (
        "weights, runtime reserve, and resident KV cache exceed HBM capacity"
    )
    assert result.peak_hbm_kv_allocated_gib == 16.03125
    assert result.hbm_capacity_headroom_gib < 0
    assert result.bottleneck == "capacity"
    assert result.to_dict(include_steps=False)["schema_version"] == "2.0"


def test_weight_and_remote_capacity_failures_are_precise() -> None:
    base = request(context_tokens=8192)
    oversized = replace(base, workload=replace(base.workload, parameter_count_b=30))
    weight_result = simulate(oversized)
    assert (
        weight_result.rejection_reason == "model weights exceed HBM capacity after runtime reserve"
    )

    constrained = replace(base, system=replace(base.system, remote_capacity_gib=1))
    remote_result = simulate(constrained)
    assert remote_result.rejection_reason == (
        "page-allocated cold KV cache exceeds remote-memory capacity"
    )
    assert remote_result.remote_capacity_headroom_gib < 0


def test_hbm_reserve_can_change_feasibility() -> None:
    base = request(context_tokens=2048)
    assert simulate(base).feasible
    constrained = replace(base, system=replace(base.system, hbm_reserved_gib=9))
    result = simulate(constrained)
    assert not result.feasible
    assert "runtime reserve" in (result.rejection_reason or "")


def test_page_rounding_affects_capacity_scan_traffic_and_fragmentation() -> None:
    result = simulate(request(context_tokens=1030, generated_tokens=1))
    step = result.steps[0]
    bytes_per_token_batch = 16 * 131_072
    assert step.cold_kv_useful_gib == pytest.approx(6 * bytes_per_token_batch / GIB)
    assert step.remote_memory_read_gib == pytest.approx(16 * bytes_per_token_batch / GIB)
    assert result.remote_memory_read_amplification == pytest.approx(16 / 6)
    assert step.hbm_kv_allocated_gib == pytest.approx(1024 * bytes_per_token_batch / GIB)
    assert step.remote_kv_allocated_gib == pytest.approx(16 * bytes_per_token_batch / GIB)
    assert result.peak_kv_fragmentation_pct > 0


def test_final_write_is_included_in_peak_allocation() -> None:
    result = simulate(request(context_tokens=1024, generated_tokens=1))
    # Read step has no cold KV, but the post-write allocation has one cold page.
    assert result.steps[0].cold_kv_useful_gib == 0
    assert result.steps[0].remote_memory_read_gib == 0
    assert result.steps[0].remote_memory_write_gib == pytest.approx(2 / 1024)
    assert result.peak_remote_kv_allocated_gib == pytest.approx(16 * 2 / 1024)


def test_remote_media_and_interconnect_are_distinct_services() -> None:
    tiered = simulate(request(context_tokens=8192))
    near = simulate(request("near_memory", context_tokens=8192))
    assert tiered.feasible and near.feasible
    assert near.total_remote_memory_read_gib == pytest.approx(tiered.total_remote_memory_read_gib)
    assert near.total_interconnect_read_gib < tiered.total_interconnect_read_gib / 1000
    assert tiered.interconnect_read_per_remote_byte == pytest.approx(1.05)
    assert 0 < near.interconnect_read_per_remote_byte < 0.001
    assert near.mean_decode_latency_ms < tiered.mean_decode_latency_ms


def test_near_memory_partial_state_and_compute_equations() -> None:
    result = simulate(request("near_memory", context_tokens=1030, generated_tokens=1))
    step = result.steps[0]
    system = request().system
    protocol = 1 + system.remote_protocol_overhead_ratio
    expected_query = 32 * 16 * 8192 * protocol / GIB
    expected_state = 32 * 16 * 32 * (128 + 2) * 4 * protocol / GIB
    physical_flops = 4 * 32 * 4096 * 16 * 16
    assert step.interconnect_write_gib == pytest.approx(
        expected_query + step.remote_memory_write_gib * protocol
    )
    assert step.interconnect_read_gib == pytest.approx(expected_state)
    assert step.near_memory_compute_ms == pytest.approx(
        physical_flops / (system.effective_near_memory_tops * 1e12) * 1000
    )


def test_peak_and_efficiency_factors_control_service_times() -> None:
    base = request(context_tokens=8192)
    result = simulate(base)
    slower_hbm = simulate(replace(base, system=replace(base.system, hbm_bandwidth_efficiency=0.5)))
    assert slower_hbm.steps[0].hbm_ms > result.steps[0].hbm_ms

    slower_link = simulate(
        replace(base, system=replace(base.system, remote_bandwidth_efficiency=0.35))
    )
    assert slower_link.steps[0].remote_link_ms == pytest.approx(result.steps[0].remote_link_ms * 2)
    slower_media = simulate(
        replace(base, system=replace(base.system, remote_memory_bandwidth_efficiency=0.35))
    )
    assert slower_media.steps[0].remote_memory_ms == pytest.approx(
        result.steps[0].remote_memory_ms * 2
    )
    slower_compute = simulate(
        replace(base, system=replace(base.system, accelerator_compute_efficiency=0.275))
    )
    assert slower_compute.steps[0].compute_ms == pytest.approx(result.steps[0].compute_ms * 2)


def test_media_can_limit_even_when_link_is_fast() -> None:
    base = request(context_tokens=8192)
    media_limited = replace(
        base,
        system=replace(
            base.system,
            remote_bandwidth_gbps=10_000,
            remote_memory_bandwidth_gbps=16,
        ),
    )
    result = simulate(media_limited)
    assert result.bottleneck == "remote_memory"
    assert result.steps[0].remote_memory_ms > result.steps[0].remote_link_ms


def test_slow_near_memory_compute_reverses_the_base_win() -> None:
    baseline = request("near_memory", context_tokens=8192)
    slow = replace(baseline, system=replace(baseline.system, near_memory_tops=0.1))
    slow_near = simulate(slow)
    tiered = simulate(request(context_tokens=8192))
    assert slow_near.mean_decode_latency_ms > tiered.mean_decode_latency_ms
    assert slow_near.bottleneck == "near_memory_compute"


def test_more_limiting_bandwidth_and_overlap_reduce_latency() -> None:
    baseline = request(context_tokens=8192)
    faster = replace(baseline, system=replace(baseline.system, remote_bandwidth_gbps=128))
    overlapped = replace(
        baseline,
        policy=replace(baseline.policy, transfer_overlap_ratio=0.9),
    )
    assert simulate(faster).mean_decode_latency_ms < simulate(baseline).mean_decode_latency_ms
    assert simulate(overlapped).mean_decode_latency_ms < simulate(baseline).mean_decode_latency_ms


def test_larger_window_reduces_cold_media_and_increases_hbm() -> None:
    small = simulate(request(window=256, context_tokens=8192))
    large = simulate(request(window=2048, context_tokens=8192))
    assert large.total_remote_memory_read_gib < small.total_remote_memory_read_gib
    assert large.peak_hbm_kv_allocated_gib > small.peak_hbm_kv_allocated_gib
    assert not simulate(request(window=8192, context_tokens=8192, generated_tokens=16)).feasible


def test_independent_service_equations_match_closed_form() -> None:
    base = request(context_tokens=8192, generated_tokens=1)
    result = simulate(base)
    step = result.steps[0]
    workload = base.workload
    system = base.system
    policy = base.policy
    cold_tokens = workload.context_tokens - policy.hbm_window_tokens
    rounded_cold_tokens = (
        (cold_tokens + policy.kv_page_tokens - 1) // policy.kv_page_tokens
    ) * policy.kv_page_tokens
    physical_cold = (
        rounded_cold_tokens
        * workload.kv_bytes_per_token_per_sequence
        * workload.concurrent_sequences
    )
    new_kv = workload.kv_bytes_per_token_per_sequence * workload.concurrent_sequences
    expected_link_ms = (
        (physical_cold + new_kv)
        * (1 + system.remote_protocol_overhead_ratio)
        / (system.effective_remote_bandwidth_gbps * 1e9)
        * 1000
    )
    expected_media_ms = (
        (physical_cold + new_kv) / (system.effective_remote_memory_bandwidth_gbps * 1e9) * 1000
    )
    expected_fixed_ms = workload.layers * system.remote_base_latency_us / 1000
    assert step.remote_link_ms == pytest.approx(expected_link_ms)
    assert step.remote_memory_ms == pytest.approx(expected_media_ms)
    assert step.remote_fixed_ms == pytest.approx(expected_fixed_ms)
    assert step.remote_service_ms == pytest.approx(
        expected_fixed_ms + max(expected_link_ms, expected_media_ms)
    )


def test_result_summary_is_consistent_and_deterministic() -> None:
    result = simulate(request("near_memory", context_tokens=8192))
    summary = result.to_dict(include_steps=False)
    assert result.p95_step_latency_ms >= result.mean_decode_latency_ms
    assert result.estimated_energy_j > 0
    assert result.total_hbm_read_gib > 0
    assert result.total_hbm_write_gib > 0
    assert result.assumptions == ASSUMPTIONS
    assert "steps" not in summary
    assert summary["total_remote_memory_read_gib"] == float(
        format(result.total_remote_memory_read_gib, ".12g")
    )
    assert len(result.to_dict(include_steps=True)["steps"]) == 3
    assert result.to_dict() == simulate(request("near_memory", context_tokens=8192)).to_dict()


def test_result_serialization_rejects_non_finite_metrics() -> None:
    result = simulate(request("near_memory", context_tokens=8192))
    corrupted = replace(result, mean_decode_latency_ms=float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        corrupted.to_dict()


def test_one_step_energy_accounts_for_each_compute_and_data_service() -> None:
    base = request(
        "near_memory",
        window=1024,
        page_tokens=16,
        context_tokens=8193,
        generated_tokens=1,
    )
    zero_energy = replace(
        base.system,
        accelerator_compute_energy_pj_per_flop=0.0,
        near_memory_compute_energy_pj_per_flop=0.0,
        hbm_energy_pj_per_byte=0.0,
        remote_memory_energy_pj_per_byte=0.0,
        remote_link_energy_pj_per_byte=0.0,
    )
    workload = base.workload
    policy = base.policy
    cold_tokens = workload.context_tokens - policy.hbm_window_tokens
    physical_cold_tokens = (
        (cold_tokens + policy.kv_page_tokens - 1) // policy.kv_page_tokens
    ) * policy.kv_page_tokens
    batch = workload.concurrent_sequences
    bytes_per_token = workload.kv_bytes_per_token_per_sequence
    new_kv_bytes = batch * bytes_per_token
    useful_cold_flops = 4 * workload.layers * workload.hidden_size * cold_tokens * batch
    physical_cold_flops = 4 * workload.layers * workload.hidden_size * physical_cold_tokens * batch
    accelerator_flops = workload.decode_flops(workload.context_tokens) - useful_cold_flops
    hbm_bytes = (
        workload.weight_bytes + policy.hbm_window_tokens * batch * bytes_per_token + new_kv_bytes
    )
    remote_media_bytes = physical_cold_tokens * batch * bytes_per_token + new_kv_bytes
    protocol = 1 + base.system.remote_protocol_overhead_ratio
    query_bytes = workload.layers * batch * workload.query_bytes_per_layer_per_sequence
    accumulator_bytes = policy.near_memory_accumulator_bits / 8
    partial_state_bytes = (
        workload.layers
        * batch
        * workload.attention_heads
        * (workload.head_dim + 2)
        * accumulator_bytes
    )
    link_bytes = (query_bytes + partial_state_bytes + new_kv_bytes) * protocol
    assert physical_cold_flops > useful_cold_flops
    assert remote_media_bytes > physical_cold_tokens * batch * bytes_per_token

    quantities = {
        "accelerator_compute_energy_pj_per_flop": accelerator_flops,
        "near_memory_compute_energy_pj_per_flop": physical_cold_flops,
        "hbm_energy_pj_per_byte": hbm_bytes,
        "remote_memory_energy_pj_per_byte": remote_media_bytes,
        "remote_link_energy_pj_per_byte": link_bytes,
    }
    for coefficient, quantity in quantities.items():
        system = replace(zero_energy, **{coefficient: 1.0})
        result = simulate(replace(base, system=system))
        assert result.estimated_energy_j == pytest.approx(quantity * 1e-12)


def test_significant_digit_serialization_preserves_tiny_nonzero_metrics() -> None:
    tiny_workload = sample_workload(
        parameter_count_b=1e-9,
        layers=1,
        hidden_size=1,
        attention_heads=1,
        kv_heads=1,
        head_dim=1,
        weight_bits=4,
        kv_bits=4,
        activation_bits=4,
        context_tokens=1,
        generated_tokens=1,
        concurrent_sequences=1,
    )
    fast_system = sample_system(
        hbm_capacity_gib=1.0,
        hbm_reserved_gib=0.0,
        hbm_bandwidth_gbps=1e15,
        remote_capacity_gib=1.0,
        remote_bandwidth_gbps=1e15,
        remote_memory_bandwidth_gbps=1e15,
        accelerator_tops=1e15,
        near_memory_tops=1e15,
        hbm_bandwidth_efficiency=1.0,
        remote_bandwidth_efficiency=1.0,
        remote_memory_bandwidth_efficiency=1.0,
        accelerator_compute_efficiency=1.0,
        near_memory_compute_efficiency=1.0,
    )
    tiny_request = SimulationRequest(
        tiny_workload,
        fast_system,
        PlacementPolicy("tiny", "hbm_only", hbm_window_tokens=1, kv_page_tokens=1),
    )
    result = simulate(tiny_request)
    serialized = result.to_dict(include_steps=False)
    assert result.mean_decode_latency_ms > 0
    assert result.estimated_energy_j > 0
    assert serialized["mean_decode_latency_ms"] > 0
    assert serialized["estimated_energy_j"] > 0
    assert serialized["throughput_tokens_s"] > 0
