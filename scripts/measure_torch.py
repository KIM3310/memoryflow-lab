from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memoryflow.io import write_json
from memoryflow.measurement import (
    bandwidth_only_model,
    compare_transfer_model,
    fit_affine_transfer_model,
)
from scripts.torch_benchmark import (
    describe_device,
    measure_copy,
    parse_mib_sizes,
    resolve_device,
    seed_torch,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "measurements" / "local-torch-copy.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure synchronized PyTorch device-copy latency and validate transfer models"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--device-label", default="")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--calibration-mib", type=parse_mib_sizes, default=(4, 16, 64))
    parser.add_argument("--validation-mib", type=parse_mib_sizes, default=(1, 8, 32))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3310)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if set(args.calibration_mib) & set(args.validation_mib):
        raise ValueError("calibration and validation sizes must be disjoint")
    if len(set(args.calibration_mib)) != len(args.calibration_mib):
        raise ValueError("calibration sizes must be unique")
    if len(set(args.validation_mib)) != len(args.validation_mib):
        raise ValueError("validation sizes must be unique")
    torch = importlib.import_module("torch")
    device = resolve_device(torch, args.device)
    if device == "cpu":
        raise RuntimeError("this evidence run requires a GPU backend; choose CUDA or MPS")

    all_sizes = tuple(dict.fromkeys(args.calibration_mib + args.validation_mib))
    seed_torch(torch, args.seed, device)
    measured = measure_copy(
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
        "schema_version": "2.0",
        "kind": "pytorch-synchronized-device-copy",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "device_backend": device,
            "device_label": describe_device(torch, device, args.device_label),
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
        "aggregation": {
            "level": "summary_statistics",
            "raw_iterations_included": False,
            "reported_statistics": ["median_ms", "p95_ms"],
        },
        "scope": {
            "measured": "PyTorch device-to-device tensor copy latency",
            "not_measured": "HBM, CXL, remote memory, near-memory/PIM, LLM end-to-end latency, "
            "or vendor products",
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
