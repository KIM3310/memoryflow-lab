from __future__ import annotations

import argparse
import random
import time
from statistics import median
from typing import Any

from memoryflow.measurement import TransferSample


def parse_mib_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated MiB values")
    return sizes


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(torch: Any, device: str, configured_label: str) -> str:
    if configured_label:
        return configured_label
    if device == "cuda":
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if device == "mps":
        return "Apple GPU (MPS)"
    return device


def seed_torch(torch: Any, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def percentile_95(values: list[float]) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return ordered[index]


def measure_copy(
    torch: Any,
    device: str,
    dtype_name: str,
    sizes_mib: tuple[int, ...],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[int, TransferSample]:
    if warmup < 1 or repeats < 5:
        raise ValueError("warmup must be >= 1 and repeats must be >= 5")
    dtype = getattr(torch, dtype_name)
    element_size = torch.empty((), dtype=dtype).element_size()
    sizes_bytes = tuple(size * 1024**2 for size in sizes_mib)
    maximum_elements = max(sizes_bytes) // element_size
    source = torch.ones(maximum_elements, dtype=dtype, device=device)
    target = torch.empty_like(source)
    views = {
        size: (source[: size // element_size], target[: size // element_size])
        for size in sizes_bytes
    }

    for size in sizes_bytes:
        source_view, target_view = views[size]
        for _ in range(warmup):
            target_view.copy_(source_view)
        synchronize(torch, device)

    timings: dict[int, list[float]] = {size: [] for size in sizes_bytes}
    randomizer = random.Random(seed)
    for _ in range(repeats):
        order = list(sizes_bytes)
        randomizer.shuffle(order)
        for size in order:
            source_view, target_view = views[size]
            synchronize(torch, device)
            started_ns = time.perf_counter_ns()
            target_view.copy_(source_view)
            synchronize(torch, device)
            timings[size].append((time.perf_counter_ns() - started_ns) / 1_000_000)

    return {
        size: TransferSample(
            size_bytes=size,
            median_ms=median(values),
            p95_ms=percentile_95(values),
        )
        for size, values in timings.items()
    }
