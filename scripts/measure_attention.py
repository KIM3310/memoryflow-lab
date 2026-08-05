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
    AttentionRooflineModel,
    AttentionSample,
    AttentionShape,
    compare_attention_affine,
    compare_attention_roofline,
    fit_affine_transfer_model,
    fit_attention_affine_model,
)
from scripts.torch_benchmark import (
    describe_device,
    measure_copy,
    parse_mib_sizes,
    percentile_95,
    resolve_device,
    seed_torch,
    synchronize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "measurements" / "local-attention.json"


def _parse_contexts(value: str) -> tuple[int, ...]:
    contexts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not contexts or any(context <= 0 for context in contexts):
        raise argparse.ArgumentTypeError("contexts must be positive comma-separated token counts")
    return contexts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure PyTorch decode attention and compare held-out analytical predictions"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--device-label", default="")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--calibration-contexts", type=_parse_contexts, default=(256, 1024, 4096))
    parser.add_argument("--validation-contexts", type=_parse_contexts, default=(512, 2048, 8192))
    parser.add_argument("--copy-calibration-mib", type=parse_mib_sizes, default=(4, 16, 64))
    parser.add_argument("--gemm-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--gemm-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3310)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _measure_gemm(
    torch: Any,
    device: str,
    dtype: Any,
    size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    if size <= 0 or warmup < 1 or repeats < 5:
        raise ValueError("GEMM size and iteration counts must be positive")
    left = torch.randn((size, size), dtype=dtype, device=device)
    right = torch.randn((size, size), dtype=dtype, device=device)
    for _ in range(warmup):
        torch.mm(left, right)
    synchronize(torch, device)

    timings: list[float] = []
    for _ in range(repeats):
        synchronize(torch, device)
        started_ns = time.perf_counter_ns()
        torch.mm(left, right)
        synchronize(torch, device)
        timings.append((time.perf_counter_ns() - started_ns) / 1_000_000)
    median_ms = median(timings)
    p95_ms = percentile_95(timings)
    flops = 2 * size**3
    return {
        "size": size,
        "flops": flops,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "effective_tops": flops / (median_ms / 1000) / 1_000_000_000_000,
    }


def _measure_attention(
    torch: Any,
    functional: Any,
    device: str,
    dtype: Any,
    shape: AttentionShape,
    contexts: tuple[int, ...],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[int, AttentionSample]:
    if warmup < 1 or repeats < 5:
        raise ValueError("warmup must be >= 1 and repeats must be >= 5")
    tensors: dict[int, tuple[Any, Any, Any]] = {}
    for context in contexts:
        query = torch.randn(
            (shape.batch_size, shape.heads, 1, shape.head_dim),
            dtype=dtype,
            device=device,
        )
        key = torch.randn(
            (shape.batch_size, shape.heads, context, shape.head_dim),
            dtype=dtype,
            device=device,
        )
        value = torch.randn_like(key)
        tensors[context] = (query, key, value)

    for context in contexts:
        query, key, value = tensors[context]
        for _ in range(warmup):
            functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        synchronize(torch, device)

    timings: dict[int, list[float]] = {context: [] for context in contexts}
    randomizer = random.Random(seed)
    for _ in range(repeats):
        order = list(contexts)
        randomizer.shuffle(order)
        for context in order:
            query, key, value = tensors[context]
            synchronize(torch, device)
            started_ns = time.perf_counter_ns()
            functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            synchronize(torch, device)
            timings[context].append((time.perf_counter_ns() - started_ns) / 1_000_000)

    return {
        context: AttentionSample(
            context_tokens=context,
            median_ms=median(values),
            p95_ms=percentile_95(values),
        )
        for context, values in timings.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    calibration_contexts = tuple(args.calibration_contexts)
    validation_contexts = tuple(args.validation_contexts)
    if set(calibration_contexts) & set(validation_contexts):
        raise ValueError("calibration and validation contexts must be disjoint")
    if len(set(calibration_contexts)) != len(calibration_contexts):
        raise ValueError("calibration contexts must be unique")
    if len(set(validation_contexts)) != len(validation_contexts):
        raise ValueError("validation contexts must be unique")

    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    device = resolve_device(torch, args.device)
    if device == "cpu":
        raise RuntimeError("this evidence run requires a GPU backend; choose CUDA or MPS")
    dtype = getattr(torch, args.dtype)
    seed_torch(torch, args.seed, device)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    shape = AttentionShape(args.batch_size, args.heads, args.head_dim, dtype_bytes)
    shape.validate()

    copy_samples_by_size = measure_copy(
        torch,
        device,
        args.dtype,
        tuple(args.copy_calibration_mib),
        args.warmup,
        args.repeats,
        args.seed,
    )
    copy_samples = tuple(copy_samples_by_size[size * 1024**2] for size in args.copy_calibration_mib)
    copy_model = fit_affine_transfer_model(copy_samples)
    gemm = _measure_gemm(
        torch,
        device,
        dtype,
        args.gemm_size,
        args.warmup,
        args.gemm_repeats,
    )

    all_contexts = tuple(dict.fromkeys(calibration_contexts + validation_contexts))
    measured = _measure_attention(
        torch,
        functional,
        device,
        dtype,
        shape,
        all_contexts,
        args.warmup,
        args.repeats,
        args.seed,
    )
    calibration = tuple(measured[context] for context in calibration_contexts)
    validation = tuple(measured[context] for context in validation_contexts)
    roofline = AttentionRooflineModel(
        compute_tops=float(gemm["effective_tops"]),
        memory_bandwidth_gbps=copy_model.bandwidth_gbps,
        base_latency_us=copy_model.base_latency_us,
    )
    attention_affine = fit_attention_affine_model(shape, calibration)

    split_by_context = {context: "calibration" for context in calibration_contexts}
    split_by_context.update({context: "validation" for context in validation_contexts})
    samples = [
        {
            "context_tokens": context,
            "split": split_by_context[context],
            "median_ms": sample.median_ms,
            "p95_ms": sample.p95_ms,
            "modeled_flops": shape.flops(context),
            "modeled_bytes": shape.modeled_bytes(context),
        }
        for context, sample in sorted(measured.items())
    ]
    copy_calibration = [
        {
            "size_bytes": sample.size_bytes,
            "median_ms": sample.median_ms,
            "p95_ms": sample.p95_ms,
            "observed_bandwidth_gbps": sample.observed_bandwidth_gbps,
        }
        for sample in copy_samples
    ]

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "pytorch-sdpa-decode-attention",
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
            "batch_size": shape.batch_size,
            "heads": shape.heads,
            "head_dim": shape.head_dim,
            "query_tokens": 1,
            "calibration_contexts": list(calibration_contexts),
            "validation_contexts": list(validation_contexts),
            "copy_calibration_sizes_mib": list(args.copy_calibration_mib),
            "gemm_size": args.gemm_size,
            "warmup_iterations": args.warmup,
            "measured_iterations_per_context": args.repeats,
            "gemm_iterations": args.gemm_repeats,
            "seed": args.seed,
            "timing_boundary": (
                "host timer around scaled_dot_product_attention with device synchronization "
                "before and after"
            ),
        },
        "scope": {
            "measured": "single-layer PyTorch scaled dot-product decode attention",
            "not_measured": (
                "model weights, multi-layer execution, KV allocation/paging, HBM, CXL, "
                "remote memory, or end-to-end serving"
            ),
            "use": (
                "compare independent roofline inputs and attention-calibrated analytical "
                "predictions on held-out context lengths"
            ),
        },
        "hardware_calibration": {
            "copy": {
                "samples": copy_calibration,
                "bandwidth_gbps": copy_model.bandwidth_gbps,
                "base_latency_us": copy_model.base_latency_us,
            },
            "gemm": gemm,
        },
        "samples": samples,
        "models": {
            "independent_roofline": {
                "compute_tops": roofline.compute_tops,
                "memory_bandwidth_gbps": roofline.memory_bandwidth_gbps,
                "base_latency_us": roofline.base_latency_us,
                "validation": compare_attention_roofline(roofline, shape, validation).to_dict(),
            },
            "attention_affine": {
                "effective_stream_gbps": attention_affine.bandwidth_gbps,
                "base_latency_us": attention_affine.base_latency_us,
                "validation": compare_attention_affine(
                    attention_affine, shape, validation
                ).to_dict(),
            },
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
