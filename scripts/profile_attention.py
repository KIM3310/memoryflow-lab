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
from memoryflow.measurement import AttentionShape
from scripts.torch_benchmark import describe_device, seed_torch, synchronize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "evidence" / "profiles" / "local-cuda-attention"


def parse_contexts(value: str) -> tuple[int, ...]:
    contexts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not contexts or any(context <= 0 for context in contexts):
        raise argparse.ArgumentTypeError("contexts must be positive comma-separated token counts")
    if len(set(contexts)) != len(contexts):
        raise argparse.ArgumentTypeError("contexts must be unique")
    return contexts


def _event_metric(event: Any, preferred: str, legacy: str) -> float:
    value = getattr(event, preferred, None)
    if value is None:
        value = getattr(event, legacy, 0.0)
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def profiler_event_to_dict(event: Any) -> dict[str, float | int | str]:
    return {
        "operation": str(event.key),
        "calls": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "device_time_total_us": _event_metric(event, "device_time_total", "cuda_time_total"),
        "self_device_time_total_us": _event_metric(
            event, "self_device_time_total", "self_cuda_time_total"
        ),
        "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
        "device_memory_usage_bytes": int(
            getattr(event, "device_memory_usage", getattr(event, "cuda_memory_usage", 0))
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a CUDA PyTorch Profiler trace for decode attention"
    )
    parser.add_argument("--device-label", default="")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--contexts", type=parse_contexts, default=(512, 2048, 8192))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--top-operations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3310)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("attention shape values must be positive")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if args.top_operations < 1:
        raise ValueError("top-operations must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_arguments(args)
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an NVIDIA GPU with CUDA-enabled PyTorch")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    seed_torch(torch, args.seed, device)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    shape = AttentionShape(args.batch_size, args.heads, args.head_dim, dtype_bytes)
    shape.validate()

    tensors: dict[int, tuple[Any, Any, Any]] = {}
    for context in args.contexts:
        query = torch.randn(
            (shape.batch_size, shape.heads, 1, shape.head_dim), dtype=dtype, device=device
        )
        key = torch.randn(
            (shape.batch_size, shape.heads, context, shape.head_dim),
            dtype=dtype,
            device=device,
        )
        tensors[context] = (query, key, torch.randn_like(key))

    for context in args.contexts:
        query, key, value = tensors[context]
        for _ in range(args.warmup):
            functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
    synchronize(torch, device)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profile:
        for repeat in range(args.repeats):
            for context in args.contexts:
                query, key, value = tensors[context]
                with torch.profiler.record_function(f"sdpa_context_{context}_repeat_{repeat}"):
                    functional.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
        synchronize(torch, device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    summary_path = output_dir / "summary.json"
    profile.export_chrome_trace(str(trace_path))

    operations = [profiler_event_to_dict(event) for event in profile.key_averages()]
    operations.sort(key=lambda item: float(item["self_device_time_total_us"]), reverse=True)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "pytorch-cuda-sdpa-profile",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "device_backend": device,
            "device_label": describe_device(torch, device, args.device_label),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_device_memory_bytes": int(properties.total_memory),
            "dtype": args.dtype,
            "machine": platform.machine(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "protocol": {
            "batch_size": shape.batch_size,
            "heads": shape.heads,
            "head_dim": shape.head_dim,
            "query_tokens": 1,
            "contexts": list(args.contexts),
            "warmup_iterations_per_context": args.warmup,
            "profiled_iterations_per_context": args.repeats,
            "seed": args.seed,
            "preallocated_kv": True,
            "synchronization": "one CUDA synchronization before and after the profiled region",
        },
        "scope": {
            "measured": "CUDA operator and kernel activity for single-layer PyTorch SDPA",
            "not_measured": (
                "model weights, multi-layer decoding, KV allocation or paging, CXL, remote "
                "memory, power, or end-to-end serving"
            ),
        },
        "trace_file": trace_path.name,
        "top_operations_by_self_device_time": operations[: args.top_operations],
    }
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
