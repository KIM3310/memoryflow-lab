from __future__ import annotations

import pytest

from memoryflow.domain import MemorySystem, PlacementPolicy, Workload


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
        "context_tokens": 4096,
        "generated_tokens": 8,
        "concurrent_sequences": 4,
    }
    values.update(overrides)
    return Workload(**values)  # type: ignore[arg-type]


def sample_system(**overrides: object) -> MemorySystem:
    values: dict[str, object] = {
        "name": "test-system",
        "hbm_capacity_gib": 24,
        "hbm_bandwidth_gbps": 900,
        "remote_capacity_gib": 128,
        "remote_bandwidth_gbps": 64,
        "remote_base_latency_us": 4,
        "accelerator_tops": 120,
    }
    values.update(overrides)
    return MemorySystem(**values)  # type: ignore[arg-type]


def test_weight_bytes_respects_quantization() -> None:
    assert sample_workload(weight_bits=16).weight_bytes == 14_000_000_000
    assert sample_workload(weight_bits=8).weight_bytes == 7_000_000_000


def test_kv_bytes_use_kv_heads_for_grouped_query_attention() -> None:
    workload = sample_workload()
    assert workload.kv_bytes_per_token_per_sequence == 131_072


def test_more_kv_heads_increase_cache_size() -> None:
    grouped = sample_workload(kv_heads=8)
    multi_head = sample_workload(kv_heads=32)
    assert multi_head.kv_bytes_per_token_per_sequence == grouped.kv_bytes_per_token_per_sequence * 4


def test_decode_flops_grow_with_sequence_length() -> None:
    workload = sample_workload()
    assert workload.decode_flops(8192) > workload.decode_flops(1024)


@pytest.mark.parametrize("field", ["layers", "context_tokens", "concurrent_sequences"])
def test_workload_rejects_non_positive_values(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sample_workload(**{field: 0}).validate()


def test_workload_rejects_more_kv_heads_than_attention_heads() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        sample_workload(kv_heads=64).validate()


def test_workload_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        sample_workload(kv_bits=3).validate()


def test_system_rejects_zero_bandwidth() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        sample_system(remote_bandwidth_gbps=0).validate()


def test_policy_rejects_invalid_overlap() -> None:
    policy = PlacementPolicy("bad", "sliding_window", 1024, transfer_overlap_ratio=1.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        policy.validate()


def test_policy_rejects_invalid_window() -> None:
    policy = PlacementPolicy("bad", "sliding_window", 0)
    with pytest.raises(ValueError, match="positive"):
        policy.validate()
