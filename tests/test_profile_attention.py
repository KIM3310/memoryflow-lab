from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts.profile_attention import parse_contexts, profiler_event_to_dict


def test_parse_contexts_accepts_positive_unique_values() -> None:
    assert parse_contexts("512, 2048,8192") == (512, 2048, 8192)


@pytest.mark.parametrize("value", ["", "0,512", "512,512"])
def test_parse_contexts_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_contexts(value)


def test_profiler_event_normalizes_device_metrics() -> None:
    event = SimpleNamespace(
        key="aten::scaled_dot_product_attention",
        count=3,
        cpu_time_total=10.0,
        self_cpu_time_total=2.0,
        device_time_total=8.0,
        self_device_time_total=7.0,
        cpu_memory_usage=128,
        device_memory_usage=256,
    )
    assert profiler_event_to_dict(event) == {
        "operation": "aten::scaled_dot_product_attention",
        "calls": 3,
        "cpu_time_total_us": 10.0,
        "self_cpu_time_total_us": 2.0,
        "device_time_total_us": 8.0,
        "self_device_time_total_us": 7.0,
        "cpu_memory_usage_bytes": 128,
        "device_memory_usage_bytes": 256,
    }
