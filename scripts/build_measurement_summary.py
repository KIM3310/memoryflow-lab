from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memoryflow.measurement import (
    AttentionRooflineModel,
    AttentionShape,
    TransferSample,
    attention_samples_from_payload,
    bandwidth_only_model,
    compare_attention_affine,
    compare_attention_roofline,
    compare_transfer_model,
    fit_affine_transfer_model,
    fit_attention_affine_model,
    samples_from_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPY_INPUT = ROOT / "evidence" / "measurements" / "apple-m4-mps-copy.json"
DEFAULT_ATTENTION_INPUT = ROOT / "evidence" / "measurements" / "apple-m4-mps-attention.json"
DEFAULT_OUTPUT = ROOT / "evidence" / "measurement-summary.md"


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"measurement payload field must be an object: {key}")
    return value


def _model_payload(payload: dict[str, Any], name: str) -> dict[str, Any]:
    models = _require_dict(payload, "models")
    value = models.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"measurement payload is missing model: {name}")
    return value


def _assert_close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"stored {label} does not match raw samples")


def _assert_validation(stored: Any, expected: dict[str, Any], label: str) -> None:
    normalized = json.loads(json.dumps(expected))
    if not _validation_matches(stored, normalized):
        raise ValueError(f"stored {label} validation does not match raw samples")


def _validation_matches(stored: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return stored is expected
    if isinstance(expected, float):
        return (
            isinstance(stored, (int, float))
            and not isinstance(stored, bool)
            and math.isfinite(float(stored))
            and math.isfinite(expected)
            and math.isclose(float(stored), expected, rel_tol=1e-9, abs_tol=1e-9)
        )
    if isinstance(expected, int):
        return isinstance(stored, int) and not isinstance(stored, bool) and stored == expected
    if isinstance(expected, dict):
        return (
            isinstance(stored, dict)
            and stored.keys() == expected.keys()
            and all(_validation_matches(stored[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(stored, list)
            and len(stored) == len(expected)
            and all(
                _validation_matches(stored_item, expected_item)
                for stored_item, expected_item in zip(stored, expected, strict=True)
            )
        )
    return stored == expected


def _validate_environment(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported measurement schema")
    environment = _require_dict(payload, "environment")
    if environment.get("device_backend") not in {"cuda", "mps"}:
        raise ValueError("committed measurement must use a GPU backend")
    return environment


def validate_copy(payload: dict[str, Any]) -> None:
    _validate_environment(payload)
    if payload.get("kind") != "pytorch-synchronized-device-copy":
        raise ValueError("unexpected copy measurement kind")
    protocol = _require_dict(payload, "protocol")
    if int(protocol["warmup_iterations_per_size"]) < 1:
        raise ValueError("copy measurement requires at least one warmup")
    if int(protocol["measured_iterations_per_size"]) < 5:
        raise ValueError("copy measurement requires at least five measured iterations")
    raw_calibration_sizes = [int(value) * 1024**2 for value in protocol["calibration_sizes_mib"]]
    raw_validation_sizes = [int(value) * 1024**2 for value in protocol["validation_sizes_mib"]]
    calibration_sizes = set(raw_calibration_sizes)
    validation_sizes = set(raw_validation_sizes)
    if len(calibration_sizes) != len(raw_calibration_sizes) or len(validation_sizes) != len(
        raw_validation_sizes
    ):
        raise ValueError("copy protocol sizes must be unique")
    if calibration_sizes & validation_sizes:
        raise ValueError("copy calibration and validation sizes must be disjoint")
    expected_split = {size: "calibration" for size in calibration_sizes}
    expected_split.update({size: "validation" for size in validation_sizes})
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("copy samples must be a list")
    seen_sizes: set[int] = set()
    for raw in raw_samples:
        if not isinstance(raw, dict):
            raise ValueError("copy sample must be an object")
        sample = TransferSample(
            size_bytes=int(raw["size_bytes"]),
            median_ms=float(raw["median_ms"]),
            p95_ms=float(raw["p95_ms"]),
        )
        sample.validate()
        if sample.size_bytes in seen_sizes:
            raise ValueError("copy sample sizes must be unique")
        seen_sizes.add(sample.size_bytes)
        if raw.get("split") != expected_split.get(sample.size_bytes):
            raise ValueError("copy sample split does not match protocol")
        _assert_close(raw.get("size_mib"), sample.size_bytes / 1024**2, "copy.size_mib")
        _assert_close(
            raw.get("observed_bandwidth_gbps"),
            sample.observed_bandwidth_gbps,
            "copy.observed_bandwidth",
        )
    if seen_sizes != set(expected_split):
        raise ValueError("copy samples do not match protocol sizes")
    calibration = samples_from_payload(payload, "calibration")
    validation = samples_from_payload(payload, "validation")
    expected_models = {
        "bandwidth_only": bandwidth_only_model(calibration),
        "affine": fit_affine_transfer_model(calibration),
    }
    for name, model in expected_models.items():
        stored = _model_payload(payload, name)
        expected_comparison = compare_transfer_model(model, validation)
        _assert_close(stored.get("bandwidth_gbps"), model.bandwidth_gbps, f"{name}.bandwidth")
        _assert_close(stored.get("base_latency_us"), model.base_latency_us, f"{name}.base")
        _assert_validation(stored.get("validation"), expected_comparison.to_dict(), f"copy.{name}")


def validate_attention(payload: dict[str, Any]) -> None:
    environment = _validate_environment(payload)
    if payload.get("kind") != "pytorch-sdpa-decode-attention":
        raise ValueError("unexpected attention measurement kind")
    protocol = _require_dict(payload, "protocol")
    if int(protocol.get("query_tokens", 0)) != 1:
        raise ValueError("attention model currently requires one query token")
    if int(protocol["warmup_iterations"]) < 1:
        raise ValueError("attention measurement requires at least one warmup")
    if int(protocol["measured_iterations_per_context"]) < 5:
        raise ValueError("attention measurement requires at least five measured iterations")
    if int(protocol["gemm_iterations"]) < 5:
        raise ValueError("attention measurement requires at least five GEMM iterations")
    dtype_bytes = {"float16": 2, "float32": 4}.get(str(environment.get("dtype")))
    if dtype_bytes is None:
        raise ValueError("unsupported attention measurement dtype")
    shape = AttentionShape(
        batch_size=int(protocol["batch_size"]),
        heads=int(protocol["heads"]),
        head_dim=int(protocol["head_dim"]),
        dtype_bytes=dtype_bytes,
    )
    calibration = attention_samples_from_payload(payload, "calibration")
    validation = attention_samples_from_payload(payload, "validation")
    raw_calibration_contexts = [int(value) for value in protocol["calibration_contexts"]]
    raw_validation_contexts = [int(value) for value in protocol["validation_contexts"]]
    calibration_contexts = set(raw_calibration_contexts)
    validation_contexts = set(raw_validation_contexts)
    if len(calibration_contexts) != len(raw_calibration_contexts) or len(
        validation_contexts
    ) != len(raw_validation_contexts):
        raise ValueError("attention protocol contexts must be unique")
    if calibration_contexts & validation_contexts:
        raise ValueError("attention calibration and validation contexts must be disjoint")
    expected_split = {context: "calibration" for context in calibration_contexts}
    expected_split.update({context: "validation" for context in validation_contexts})
    raw_attention_samples = payload.get("samples")
    if not isinstance(raw_attention_samples, list):
        raise ValueError("attention samples must be a list")
    seen_contexts: set[int] = set()
    for raw in raw_attention_samples:
        if not isinstance(raw, dict):
            raise ValueError("attention sample must be an object")
        context = int(raw["context_tokens"])
        if context in seen_contexts:
            raise ValueError("attention contexts must be unique")
        seen_contexts.add(context)
        if raw.get("split") != expected_split.get(context):
            raise ValueError("attention sample split does not match protocol")
        _assert_close(raw.get("modeled_flops"), float(shape.flops(context)), "attention.flops")
        _assert_close(
            raw.get("modeled_bytes"), float(shape.modeled_bytes(context)), "attention.bytes"
        )
    if seen_contexts != set(expected_split):
        raise ValueError("attention samples do not match protocol contexts")

    hardware = _require_dict(payload, "hardware_calibration")
    copy_payload = hardware.get("copy")
    gemm = hardware.get("gemm")
    if not isinstance(copy_payload, dict) or not isinstance(gemm, dict):
        raise ValueError("attention hardware calibration is incomplete")
    raw_copy_samples = copy_payload.get("samples")
    if not isinstance(raw_copy_samples, list):
        raise ValueError("attention copy calibration samples must be a list")
    copy_samples = tuple(
        TransferSample(
            size_bytes=int(sample["size_bytes"]),
            median_ms=float(sample["median_ms"]),
            p95_ms=float(sample["p95_ms"]),
        )
        for sample in raw_copy_samples
        if isinstance(sample, dict)
    )
    expected_copy_sizes = {int(value) * 1024**2 for value in protocol["copy_calibration_sizes_mib"]}
    if {sample.size_bytes for sample in copy_samples} != expected_copy_sizes:
        raise ValueError("attention copy samples do not match protocol sizes")
    for raw, sample in zip(raw_copy_samples, copy_samples, strict=True):
        if not isinstance(raw, dict):
            raise ValueError("attention copy sample must be an object")
        sample.validate()
        _assert_close(
            raw.get("observed_bandwidth_gbps"),
            sample.observed_bandwidth_gbps,
            "attention.copy.observed_bandwidth",
        )
    copy_model = fit_affine_transfer_model(copy_samples)
    _assert_close(
        copy_payload.get("bandwidth_gbps"), copy_model.bandwidth_gbps, "attention.copy.bandwidth"
    )
    _assert_close(
        copy_payload.get("base_latency_us"), copy_model.base_latency_us, "attention.copy.base"
    )

    gemm_size = int(gemm["size"])
    if gemm_size != int(protocol["gemm_size"]):
        raise ValueError("stored GEMM size does not match protocol")
    gemm_flops = 2 * gemm_size**3
    if int(gemm["flops"]) != gemm_flops:
        raise ValueError("stored GEMM FLOPs do not match GEMM size")
    gemm_median_ms = float(gemm["median_ms"])
    gemm_p95_ms = float(gemm["p95_ms"])
    if gemm_median_ms <= 0 or gemm_p95_ms < gemm_median_ms:
        raise ValueError("stored GEMM latency statistics are invalid")
    compute_tops = gemm_flops / (gemm_median_ms / 1000) / 1_000_000_000_000
    _assert_close(gemm.get("effective_tops"), compute_tops, "attention.gemm.effective_tops")

    roofline = AttentionRooflineModel(
        compute_tops=compute_tops,
        memory_bandwidth_gbps=copy_model.bandwidth_gbps,
        base_latency_us=copy_model.base_latency_us,
    )
    attention_affine = fit_attention_affine_model(shape, calibration)
    roofline_stored = _model_payload(payload, "independent_roofline")
    affine_stored = _model_payload(payload, "attention_affine")
    _assert_close(roofline_stored.get("compute_tops"), compute_tops, "roofline.compute_tops")
    _assert_close(
        roofline_stored.get("memory_bandwidth_gbps"),
        copy_model.bandwidth_gbps,
        "roofline.bandwidth",
    )
    _assert_close(
        roofline_stored.get("base_latency_us"), copy_model.base_latency_us, "roofline.base"
    )
    _assert_close(
        affine_stored.get("effective_stream_gbps"),
        attention_affine.bandwidth_gbps,
        "attention_affine.stream_rate",
    )
    _assert_close(
        affine_stored.get("base_latency_us"),
        attention_affine.base_latency_us,
        "attention_affine.base",
    )
    _assert_validation(
        roofline_stored.get("validation"),
        compare_attention_roofline(roofline, shape, validation).to_dict(),
        "attention.independent_roofline",
    )
    _assert_validation(
        affine_stored.get("validation"),
        compare_attention_affine(attention_affine, shape, validation).to_dict(),
        "attention.attention_affine",
    )


def render(copy_payload: dict[str, Any], attention_payload: dict[str, Any]) -> str:
    validate_copy(copy_payload)
    validate_attention(attention_payload)
    copy_environment = copy_payload["environment"]
    attention_environment = attention_payload["environment"]
    for field in (
        "device_backend",
        "device_label",
        "dtype",
        "torch_version",
        "machine",
        "os",
        "python_version",
    ):
        if copy_environment[field] != attention_environment[field]:
            raise ValueError(f"copy and attention environments differ: {field}")

    attention_protocol = attention_payload["protocol"]
    attention_samples = attention_payload["samples"]
    roofline = _model_payload(attention_payload, "independent_roofline")
    attention_affine = _model_payload(attention_payload, "attention_affine")
    roofline_validation = roofline["validation"]
    attention_validation = attention_affine["validation"]
    roofline_points = {point["context_tokens"]: point for point in roofline_validation["points"]}
    attention_points = {point["context_tokens"]: point for point in attention_validation["points"]}

    copy_samples = copy_payload["samples"]
    bandwidth_only = _model_payload(copy_payload, "bandwidth_only")
    copy_affine = _model_payload(copy_payload, "affine")
    copy_protocol = copy_payload["protocol"]

    lines = [
        "# PyTorch GPU Measurement Summary",
        "",
        "## Environment",
        "",
        f"- Device: `{attention_environment['device_label']}`",
        f"- Backend: `{attention_environment['device_backend']}`",
        f"- PyTorch: `{attention_environment['torch_version']}`",
        f"- Dtype: `{attention_environment['dtype']}`",
        f"- Attention run: `{attention_payload['generated_at_utc']}`",
        f"- Copy run: `{copy_payload['generated_at_utc']}`",
        "",
        "## Decode attention measurement",
        "",
        (
            f"PyTorch SDPA shape: batch {attention_protocol['batch_size']}, "
            f"{attention_protocol['heads']} heads, head dimension "
            f"{attention_protocol['head_dim']}, one query token."
        ),
        "",
        "| Split | Context | Median (ms) | p95 (ms) | Modeled bytes (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in attention_samples:
        lines.append(
            f"| {sample['split']} | {int(sample['context_tokens']):,} | "
            f"{float(sample['median_ms']):.4f} | {float(sample['p95_ms']):.4f} | "
            f"{float(sample['modeled_bytes']) / 1024**2:.2f} |"
        )
    lines.extend(
        [
            "",
            "Calibration contexts are disjoint from validation contexts.",
            "",
            "| Model | Validation MAPE | Max error |",
            "|---|---:|---:|",
            (
                "| independent copy/GEMM roofline | "
                f"{float(roofline_validation['mean_absolute_percentage_error_pct']):.2f}% | "
                f"{float(roofline_validation['max_absolute_percentage_error_pct']):.2f}% |"
            ),
            (
                "| attention-calibrated affine | "
                f"{float(attention_validation['mean_absolute_percentage_error_pct']):.2f}% | "
                f"{float(attention_validation['max_absolute_percentage_error_pct']):.2f}% |"
            ),
            "",
            "| Validation context | Measured (ms) | Roofline (ms) | Calibrated (ms) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for context in attention_protocol["validation_contexts"]:
        roofline_point = roofline_points[context]
        attention_point = attention_points[context]
        lines.append(
            f"| {int(context):,} | {float(attention_point['measured_ms']):.4f} | "
            f"{float(roofline_point['predicted_ms']):.4f} | "
            f"{float(attention_point['predicted_ms']):.4f} |"
        )

    calibration_sizes = ", ".join(str(value) for value in copy_protocol["calibration_sizes_mib"])
    validation_sizes = ", ".join(str(value) for value in copy_protocol["validation_sizes_mib"])
    lines.extend(
        [
            "",
            "## Supporting device-copy measurement",
            "",
            (
                f"Calibration uses {calibration_sizes} MiB transfers. Validation uses separate "
                f"{validation_sizes} MiB transfers."
            ),
            "",
            "| Split | Size (MiB) | Median (ms) | p95 (ms) | Observed GB/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sample in copy_samples:
        lines.append(
            f"| {sample['split']} | {float(sample['size_mib']):.0f} | "
            f"{float(sample['median_ms']):.4f} | {float(sample['p95_ms']):.4f} | "
            f"{float(sample['observed_bandwidth_gbps']):.2f} |"
        )
    lines.extend(
        [
            "",
            "| Transfer model | Validation MAPE | Max error |",
            "|---|---:|---:|",
            (
                "| bytes / bandwidth | "
                f"{float(bandwidth_only['validation']['mean_absolute_percentage_error_pct']):.2f}% "
                "| "
                f"{float(bandwidth_only['validation']['max_absolute_percentage_error_pct']):.2f}% |"
            ),
            (
                "| base latency + bytes / bandwidth | "
                f"{float(copy_affine['validation']['mean_absolute_percentage_error_pct']):.2f}% | "
                f"{float(copy_affine['validation']['max_absolute_percentage_error_pct']):.2f}% |"
            ),
            "",
            "## Boundary",
            "",
            (
                "The attention run measures one fused PyTorch SDPA layer with preallocated KV "
                "tensors. It excludes model weights, multi-layer execution, KV allocation and "
                "paging, HBM, CXL, remote memory, and serving orchestration. The copy run "
                "validates the fixed-latency transfer equation on the recorded GPU backend. "
                "Neither result is a named-product performance claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and render committed GPU measurements")
    parser.add_argument("--copy-input", type=Path, default=DEFAULT_COPY_INPUT)
    parser.add_argument("--attention-input", type=Path, default=DEFAULT_ATTENTION_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    copy_payload = json.loads(args.copy_input.read_text(encoding="utf-8"))
    attention_payload = json.loads(args.attention_input.read_text(encoding="utf-8"))
    if not isinstance(copy_payload, dict) or not isinstance(attention_payload, dict):
        raise ValueError("measurement payload roots must be objects")
    rendered = render(copy_payload, attention_payload)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"measurement summary is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
