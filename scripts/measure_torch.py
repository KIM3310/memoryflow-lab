from __future__ import annotations

import argparse
import importlib
import json
import platform
import random
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from memoryflow.io import write_json
from memoryflow.measurement import (
    TransferSample,
    bandwidth_only_model,
    compare_transfer_model,
    fit_affine_transfer_model,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "measurements" / "local-torch-copy.json"


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated MiB values")
    return sizes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure synchronized PyTorch device-copy latency and validate transfer models"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--device-label", default="")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--calibration-mib", type=_parse_sizes, default=(4, 16, 64))
    parser.add_argument("--validation-mib", type=_parse_sizes, default=(1, 8, 32))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3310)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _resolve_device(torch: Any, requested: str) -> str:
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


def _synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return ordered[index]


def _measure(
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
        _synchronize(torch, device)

    timings: dict[int, list[float]] = {size: [] for size in sizes_bytes}
    randomizer = random.Random(seed)
    for _ in range(repeats):
        order = list(sizes_bytes)
        randomizer.shuffle(order)
        for size in order:
            source_view, target_view = views[size]
            _synchronize(torch, device)
            started_ns = time.perf_counter_ns()
            target_view.copy_(source_view)
            _synchronize(torch, device)
            timings[size].append((time.perf_counter_ns() - started_ns) / 1_000_000)

    return {
        size: TransferSample(
            size_bytes=size,
            median_ms=median(values),
            p95_ms=_percentile_95(values),
        )
        for size, values in timings.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if set(args.calibration_mib) & set(args.validation_mib):
        raise ValueError("calibration and validation sizes must be disjoint")
    if len(set(args.calibration_mib)) != len(args.calibration_mib):
        raise ValueError("calibration sizes must be unique")
    if len(set(args.validation_mib)) != len(args.validation_mib):
        raise ValueError("validation sizes must be unique")
    torch = importlib.import_module("torch")
    device = _resolve_device(torch, args.device)
    if device == "cpu":
        raise RuntimeError("this evidence run requires a GPU backend; choose CUDA or MPS")

    all_sizes = tuple(dict.fromkeys(args.calibration_mib + args.validation_mib))
    measured = _measure(
        torch,
        device,
        args.dtype,
        all_sizes,
        args.warmup,
        args.repeats,
        args.seed,
    )
    calibration = tuple(measured[size * 1024**2] for size in args.calibration_mib)
    validation = tuple(measured[size * 1024**2] for size in args.validation_mib)
    bandwidth_only = bandwidth_only_model(calibration)
    affine = fit_affine_transfer_model(calibration)

    split_by_size = {size * 1024**2: "calibration" for size in args.calibration_mib}
    split_by_size.update({size * 1024**2: "validation" for size in args.validation_mib})
    samples = []
    for size, sample in sorted(measured.items()):
        samples.append(
            {
                "size_bytes": size,
                "size_mib": size / 1024**2,
                "split": split_by_size[size],
                "median_ms": sample.median_ms,
                "p95_ms": sample.p95_ms,
                "observed_bandwidth_gbps": sample.observed_bandwidth_gbps,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "pytorch-synchronized-device-copy",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "device_backend": device,
            "device_label": args.device_label or device,
            "dtype": args.dtype,
            "machine": platform.machine(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "protocol": {
            "warmup_iterations_per_size": args.warmup,
            "measured_iterations_per_size": args.repeats,
            "seed": args.seed,
            "timing_boundary": (
                "host timer around copy_ with device synchronization before and after"
            ),
            "calibration_sizes_mib": list(args.calibration_mib),
            "validation_sizes_mib": list(args.validation_mib),
        },
        "scope": {
            "measured": "PyTorch device-to-device tensor copy latency",
            "not_measured": "HBM, CXL, remote memory, LLM end-to-end latency, or vendor products",
            "use": "validate the bytes/bandwidth plus fixed-latency equation shape",
        },
        "samples": samples,
        "models": {
            "bandwidth_only": {
                "bandwidth_gbps": bandwidth_only.bandwidth_gbps,
                "base_latency_us": bandwidth_only.base_latency_us,
                "validation": compare_transfer_model(bandwidth_only, validation).to_dict(),
            },
            "affine": {
                "bandwidth_gbps": affine.bandwidth_gbps,
                "base_latency_us": affine.base_latency_us,
                "validation": compare_transfer_model(affine, validation).to_dict(),
            },
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
