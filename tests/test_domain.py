from __future__ import annotations

import pytest

from memoryflow.domain import (
    MAX_CONCURRENT_SEQUENCES,
    MAX_CONTEXT_TOKENS,
    MAX_GENERATED_TOKENS,
    MemorySystem,
    PlacementPolicy,
    ScenarioProvenance,
    Workload,
)


def sample_workload(**overrides: object) -> Workload:
    values: dict[str, object] = {
        "name": "test",
        "parameter_count_b": 7.0,
        "layers": 32,
        "hidden_size": 4096,
        "attention_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "weight_bits": 16,
        "kv_bits": 16,
        "activation_bits": 16,
        "context_tokens": 4096,
        "generated_tokens": 8,
        "concurrent_sequences": 4,
    }
    values.update(overrides)
    return Workload(**values)  # type: ignore[arg-type]


def sample_system(**overrides: object) -> MemorySystem:
    values: dict[str, object] = {
        "name": "test-system",
        "hbm_capacity_gib": 24.0,
        "hbm_bandwidth_gbps": 900.0,
        "remote_capacity_gib": 128.0,
        "remote_bandwidth_gbps": 64.0,
        "remote_memory_bandwidth_gbps": 128.0,
        "remote_base_latency_us": 4.0,
        "accelerator_tops": 120.0,
    }
    values.update(overrides)
    return MemorySystem(**values)  # type: ignore[arg-type]


def test_byte_and_flop_equations() -> None:
    workload = sample_workload()
    assert workload.weight_bytes == 14_000_000_000
    assert sample_workload(weight_bits=8).weight_bytes == 7_000_000_000
    assert workload.kv_bytes_per_layer_per_token_per_sequence == 4096
    assert workload.kv_bytes_per_token_per_sequence == 131_072
    assert workload.query_bytes_per_layer_per_sequence == 8192
    assert sample_workload(kv_heads=32).kv_bytes_per_token_per_sequence == 4 * (
        workload.kv_bytes_per_token_per_sequence
    )
    assert workload.decode_flops(8192) > workload.decode_flops(1024)


@pytest.mark.parametrize("field", ["layers", "context_tokens", "concurrent_sequences"])
def test_workload_rejects_non_positive_values(field: str) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        sample_workload(**{field: 0}).validate()


@pytest.mark.parametrize("value", [1.5, True])
def test_workload_rejects_non_integer_dimensions(value: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        sample_workload(layers=value).validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000, True])
def test_workload_rejects_invalid_parameter_counts(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        sample_workload(parameter_count_b=value).validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("context_tokens", MAX_CONTEXT_TOKENS + 1, "cannot exceed"),
        ("generated_tokens", MAX_GENERATED_TOKENS + 1, "cannot exceed"),
        ("concurrent_sequences", MAX_CONCURRENT_SEQUENCES + 1, "cannot exceed"),
        ("kv_heads", 64, "cannot exceed"),
        ("kv_heads", 7, "divisible"),
        ("hidden_size", 4095, r"attention_heads \* head_dim"),
        ("kv_bits", 3, "must be one of"),
        ("activation_bits", 3, "must be one of"),
        ("parameter_count_b", 100_001, "cannot exceed"),
        ("layers", 4_097, "supported bounds"),
        ("name", " ", "non-empty string"),
        ("name", "\ud800", "valid Unicode scalar values"),
    ],
)
def test_workload_rejects_invalid_contracts(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_workload(**{field: value}).validate()


def test_decode_flops_requires_positive_integer_length() -> None:
    for value in (0, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            sample_workload().decode_flops(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("remote_bandwidth_gbps", 0, "positive finite"),
        ("hbm_bandwidth_gbps", 5e-324, "effective rates"),
        ("near_memory_compute_efficiency", 5e-324, "effective rates"),
        ("remote_memory_bandwidth_gbps", float("nan"), "positive finite"),
        ("hbm_reserved_gib", -1, "non-negative finite"),
        ("remote_base_latency_us", float("inf"), "non-negative finite"),
        ("hbm_reserved_gib", 24, "lower than"),
        ("hbm_bandwidth_efficiency", 0, "efficiency"),
        ("remote_bandwidth_efficiency", 1.1, "efficiency"),
        ("accelerator_compute_efficiency", True, "efficiency"),
        ("remote_protocol_overhead_ratio", -0.1, "in \[0, 1\]"),
        ("remote_protocol_overhead_ratio", float("nan"), "in \[0, 1\]"),
        ("remote_capacity_gib", 1e16, "supported bounds"),
        ("name", "", "non-empty string"),
    ],
)
def test_system_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_system(**{field: value}).validate()


def test_effective_system_rates_and_capacity_are_explicit() -> None:
    system = sample_system(
        hbm_reserved_gib=2,
        hbm_bandwidth_efficiency=0.5,
        remote_bandwidth_efficiency=0.25,
        remote_memory_bandwidth_efficiency=0.75,
        accelerator_compute_efficiency=0.4,
        near_memory_compute_efficiency=0.3,
    )
    system.validate()
    assert system.usable_hbm_capacity_bytes == (24 - 2) * 1024**3
    assert system.effective_hbm_bandwidth_gbps == 450
    assert system.effective_remote_bandwidth_gbps == 16
    assert system.effective_remote_memory_bandwidth_gbps == 96
    assert system.effective_accelerator_tops == 48
    assert system.effective_near_memory_tops == pytest.approx(3.6)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (PlacementPolicy("bad", "sliding_window", 0), "positive integer"),
        (PlacementPolicy("bad", "sliding_window", 1025), "divisible"),
        (PlacementPolicy("bad", "sliding_window", 1024, transfer_overlap_ratio=1.1), "between"),
        (
            PlacementPolicy("bad", "sliding_window", 1024, transfer_overlap_ratio=float("nan")),
            "finite number",
        ),
        (
            PlacementPolicy("bad", "sliding_window", 1024, near_memory_accumulator_bits=8),
            "16 or 32",
        ),
        (PlacementPolicy("bad", "sliding_window", 1024, kv_page_tokens=True), "positive integer"),
        (PlacementPolicy("bad", "sliding_window", 4_198_401), "cannot exceed"),
        (
            PlacementPolicy("bad", "sliding_window", 131_072, kv_page_tokens=131_072),
            "cannot exceed",
        ),
    ],
)
def test_policy_validation(policy: PlacementPolicy, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        policy.validate()


def test_provenance_validation_preserves_evidence_boundary() -> None:
    ScenarioProvenance("synthetic", "illustrative", "none").validate()
    ScenarioProvenance("user_supplied", "operator input", "user_supplied").validate()
    with pytest.raises(ValueError, match="measurement_scope"):
        ScenarioProvenance("synthetic", "bad", "user_supplied").validate()
    with pytest.raises(ValueError, match="hardware_profile"):
        ScenarioProvenance("measured", "bad", "none").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        ScenarioProvenance("synthetic", "", "none").validate()
