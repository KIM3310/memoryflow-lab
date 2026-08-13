from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
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


COPY_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "generated_at_utc",
    "environment",
    "protocol",
    "scope",
    "aggregation",
    "samples",
    "models",
}
ATTENTION_TOP_LEVEL_KEYS = COPY_TOP_LEVEL_KEYS | {"hardware_calibration"}
ENVIRONMENT_KEYS = {
    "device_backend",
    "device_label",
    "dtype",
    "torch_version",
    "machine",
    "os",
    "python_version",
}
AGGREGATION_KEYS = {"level", "raw_iterations_included", "reported_statistics"}
SCOPE_KEYS = {"measured", "not_measured", "use"}


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append(f"unknown={','.join(unknown)}")
        if missing:
            details.append(f"missing={','.join(missing)}")
        raise ValueError(f"{label} schema mismatch ({'; '.join(details)})")


def _validate_aggregation(payload: dict[str, Any]) -> None:
    aggregation = _require_dict(payload, "aggregation")
    _expect_keys(aggregation, AGGREGATION_KEYS, "aggregation")
    if aggregation.get("level") != "summary_statistics":
        raise ValueError("measurement artifacts must contain summary statistics")
    if aggregation.get("raw_iterations_included") is not False:
        raise ValueError("committed measurement artifacts must not contain raw iterations")
    if aggregation.get("reported_statistics") != ["median_ms", "p95_ms"]:
        raise ValueError("measurement aggregation statistics must be median_ms and p95_ms")


def _validate_generated_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("generated_at_utc must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at_utc must include a UTC offset")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


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
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError(f"stored {label} does not match aggregate samples")


def _assert_validation(stored: Any, expected: dict[str, Any], label: str) -> None:
    normalized = json.loads(json.dumps(expected))
    if not _validation_matches(stored, normalized):
        raise ValueError(f"stored {label} validation does not match aggregate samples")


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
    return bool(stored == expected)


def _validate_environment(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "2.0":
        raise ValueError("unsupported measurement schema")
    _validate_generated_at(payload.get("generated_at_utc"))
    environment = _require_dict(payload, "environment")
    _expect_keys(environment, ENVIRONMENT_KEYS, "environment")
    if environment.get("device_backend") not in {"cuda", "mps"}:
        raise ValueError("measurement must use a GPU backend")
    for key in ENVIRONMENT_KEYS:
        if not isinstance(environment.get(key), str) or not str(environment[key]).strip():
            raise ValueError(f"measurement environment field must be a non-empty string: {key}")
    scope = _require_dict(payload, "scope")
    _expect_keys(scope, SCOPE_KEYS, "scope")
    if any(
        not isinstance(scope.get(key), str) or not str(scope[key]).strip() for key in SCOPE_KEYS
    ):
        raise ValueError("measurement scope fields must be non-empty strings")
    _validate_aggregation(payload)
    return environment


def validate_copy(payload: dict[str, Any]) -> None:
    _expect_keys(payload, COPY_TOP_LEVEL_KEYS, "copy measurement")
    _validate_environment(payload)
    if payload.get("kind") != "pytorch-synchronized-device-copy":
        raise ValueError("unexpected copy measurement kind")
    protocol = _require_dict(payload, "protocol")
    _expect_keys(
        protocol,
        {
            "warmup_iterations_per_size",
            "measured_iterations_per_size",
            "seed",
            "timing_boundary",
            "calibration_sizes_mib",
            "validation_sizes_mib",
        },
        "copy protocol",
    )
    if _integer(protocol["warmup_iterations_per_size"], "copy warmup iterations") < 1:
        raise ValueError("copy measurement requires at least one warmup")
    if _integer(protocol["measured_iterations_per_size"], "copy measured iterations") < 5:
        raise ValueError("copy measurement requires at least five measured iterations")
    if _integer(protocol["seed"], "copy seed") < 0:
        raise ValueError("copy seed must be non-negative")
    _non_empty_string(protocol["timing_boundary"], "copy timing_boundary")
    raw_calibration_sizes = [
        _integer(value, "copy calibration size") * 1024**2
        for value in protocol["calibration_sizes_mib"]
    ]
    raw_validation_sizes = [
        _integer(value, "copy validation size") * 1024**2
        for value in protocol["validation_sizes_mib"]
    ]
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
        _expect_keys(
            raw,
            {"size_bytes", "size_mib", "split", "median_ms", "p95_ms", "observed_bandwidth_gbps"},
            "copy sample",
        )
        sample = TransferSample(
            size_bytes=_integer(raw["size_bytes"], "copy sample size_bytes"),
            median_ms=_number(raw["median_ms"], "sample median_ms"),
            p95_ms=_number(raw["p95_ms"], "sample p95_ms"),
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
    models = _require_dict(payload, "models")
    _expect_keys(models, {"bandwidth_only", "affine"}, "copy models")
    expected_models = {
        "bandwidth_only": bandwidth_only_model(calibration),
        "affine": fit_affine_transfer_model(calibration),
    }
    for name, model in expected_models.items():
        stored = _model_payload(payload, name)
        _expect_keys(
            stored, {"bandwidth_gbps", "base_latency_us", "validation"}, f"copy model {name}"
        )
        expected_comparison = compare_transfer_model(model, validation)
        _assert_close(stored.get("bandwidth_gbps"), model.bandwidth_gbps, f"{name}.bandwidth")
        _assert_close(stored.get("base_latency_us"), model.base_latency_us, f"{name}.base")
        _assert_validation(stored.get("validation"), expected_comparison.to_dict(), f"copy.{name}")


def validate_attention(payload: dict[str, Any]) -> None:
    _expect_keys(payload, ATTENTION_TOP_LEVEL_KEYS, "attention measurement")
    environment = _validate_environment(payload)
    if payload.get("kind") != "pytorch-sdpa-decode-attention":
        raise ValueError("unexpected attention measurement kind")
    protocol = _require_dict(payload, "protocol")
    _expect_keys(
        protocol,
        {
            "batch_size",
            "heads",
            "head_dim",
            "query_tokens",
            "calibration_contexts",
            "validation_contexts",
            "copy_calibration_sizes_mib",
            "gemm_size",
            "warmup_iterations",
            "measured_iterations_per_context",
            "gemm_iterations",
            "seed",
            "timing_boundary",
        },
        "attention protocol",
    )
    if _integer(protocol.get("query_tokens", 0), "query_tokens") != 1:
        raise ValueError("attention model currently requires one query token")
    if _integer(protocol["warmup_iterations"], "attention warmup iterations") < 1:
        raise ValueError("attention measurement requires at least one warmup")
    if _integer(protocol["measured_iterations_per_context"], "attention measured iterations") < 5:
        raise ValueError("attention measurement requires at least five measured iterations")
    if _integer(protocol["gemm_iterations"], "GEMM iterations") < 5:
        raise ValueError("attention measurement requires at least five GEMM iterations")
    if _integer(protocol["seed"], "attention seed") < 0:
        raise ValueError("attention seed must be non-negative")
    _non_empty_string(protocol["timing_boundary"], "attention timing_boundary")
    dtype_bytes = {"float16": 2, "float32": 4}.get(str(environment.get("dtype")))
    if dtype_bytes is None:
        raise ValueError("unsupported attention measurement dtype")
    shape = AttentionShape(
        batch_size=_integer(protocol["batch_size"], "attention batch_size"),
        heads=_integer(protocol["heads"], "attention heads"),
        head_dim=_integer(protocol["head_dim"], "attention head_dim"),
        dtype_bytes=dtype_bytes,
    )
    raw_calibration_contexts = [
        _integer(value, "attention calibration context")
        for value in protocol["calibration_contexts"]
    ]
    raw_validation_contexts = [
        _integer(value, "attention validation context") for value in protocol["validation_contexts"]
    ]
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
        _expect_keys(
            raw,
            {"context_tokens", "split", "median_ms", "p95_ms", "modeled_flops", "modeled_bytes"},
            "attention sample",
        )
        context = _integer(raw["context_tokens"], "attention sample context")
        median_ms = _number(raw["median_ms"], "attention sample median_ms")
        p95_ms = _number(raw["p95_ms"], "attention sample p95_ms")
        if median_ms <= 0 or p95_ms < median_ms:
            raise ValueError("attention sample latency statistics are invalid")
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
    calibration = attention_samples_from_payload(payload, "calibration")
    validation = attention_samples_from_payload(payload, "validation")

    hardware = _require_dict(payload, "hardware_calibration")
    _expect_keys(hardware, {"copy", "gemm"}, "attention hardware calibration")
    copy_payload = hardware.get("copy")
    gemm = hardware.get("gemm")
    if not isinstance(copy_payload, dict) or not isinstance(gemm, dict):
        raise ValueError("attention hardware calibration is incomplete")
    _expect_keys(
        copy_payload, {"samples", "bandwidth_gbps", "base_latency_us"}, "attention copy calibration"
    )
    _expect_keys(
        gemm,
        {"size", "flops", "median_ms", "p95_ms", "effective_tops"},
        "attention GEMM calibration",
    )
    raw_copy_samples = copy_payload.get("samples")
    if not isinstance(raw_copy_samples, list):
        raise ValueError("attention copy calibration samples must be a list")
    for raw in raw_copy_samples:
        if not isinstance(raw, dict):
            raise ValueError("attention copy sample must be an object")
        _expect_keys(
            raw,
            {"size_bytes", "median_ms", "p95_ms", "observed_bandwidth_gbps"},
            "attention copy sample",
        )
    copy_samples = tuple(
        TransferSample(
            size_bytes=_integer(sample["size_bytes"], "attention copy size_bytes"),
            median_ms=_number(sample["median_ms"], "attention copy median_ms"),
            p95_ms=_number(sample["p95_ms"], "attention copy p95_ms"),
        )
        for sample in raw_copy_samples
        if isinstance(sample, dict)
    )
    raw_expected_copy_sizes = [
        _integer(value, "attention copy calibration size") * 1024**2
        for value in protocol["copy_calibration_sizes_mib"]
    ]
    expected_copy_sizes = set(raw_expected_copy_sizes)
    if len(expected_copy_sizes) != len(raw_expected_copy_sizes):
        raise ValueError("attention copy calibration sizes must be unique")
    observed_copy_sizes = [sample.size_bytes for sample in copy_samples]
    if len(set(observed_copy_sizes)) != len(observed_copy_sizes):
        raise ValueError("attention copy sample sizes must be unique")
    if set(observed_copy_sizes) != expected_copy_sizes:
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

    gemm_size = _integer(gemm["size"], "GEMM size")
    if gemm_size != _integer(protocol["gemm_size"], "protocol GEMM size"):
        raise ValueError("stored GEMM size does not match protocol")
    gemm_flops = 2 * gemm_size**3
    if _integer(gemm["flops"], "GEMM FLOPs") != gemm_flops:
        raise ValueError("stored GEMM FLOPs do not match GEMM size")
    gemm_median_ms = _number(gemm["median_ms"], "GEMM median_ms")
    gemm_p95_ms = _number(gemm["p95_ms"], "GEMM p95_ms")
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
    models = _require_dict(payload, "models")
    _expect_keys(models, {"independent_roofline", "attention_affine"}, "attention models")
    roofline_stored = _model_payload(payload, "independent_roofline")
    affine_stored = _model_payload(payload, "attention_affine")
    _expect_keys(
        roofline_stored,
        {"compute_tops", "memory_bandwidth_gbps", "base_latency_us", "validation"},
        "independent roofline model",
    )
    _expect_keys(
        affine_stored,
        {"effective_stream_gbps", "base_latency_us", "validation"},
        "attention affine model",
    )
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


def validate_committed_apple_m4_pair(
    copy_payload: dict[str, Any], attention_payload: dict[str, Any]
) -> None:
    validate_copy(copy_payload)
    validate_attention(attention_payload)
    copy_environment = _require_dict(copy_payload, "environment")
    attention_environment = _require_dict(attention_payload, "environment")
    if copy_environment != attention_environment:
        raise ValueError("committed Apple copy and attention environments must match exactly")
    if copy_environment.get("device_backend") != "mps" or "Apple M4" not in str(
        copy_environment.get("device_label")
    ):
        raise ValueError("committed reference artifacts must identify the Apple M4 MPS run")
    for payload in (copy_payload, attention_payload):
        scope = _require_dict(payload, "scope")
        boundary = str(scope["not_measured"]).lower()
        for excluded in ("hbm", "cxl", "remote memory", "near-memory/pim", "end-to-end"):
            if excluded not in boundary:
                raise ValueError(f"committed measurement boundary must exclude {excluded}")


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

    device_label = str(attention_environment["device_label"])
    backend = str(attention_environment["device_backend"]).upper()
    other_backends = "CUDA or other backends" if backend == "MPS" else "MPS or other backends"
    attention_scope = _require_dict(attention_payload, "scope")
    copy_scope = _require_dict(copy_payload, "scope")

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
        "# PyTorch GPU Aggregate Measurement Summary",
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
                f"The {device_label} {backend} artifacts contain median/p95 aggregate summaries, "
                "not raw iterations. The attention run measures one fused PyTorch SDPA layer "
                "with preallocated KV tensors; its declared exclusions are: "
                f"{attention_scope['not_measured']}. "
                f"The copy run checks the fixed-latency transfer equation shape on {backend}; "
                f"its declared exclusions are: {copy_scope['not_measured']}. These artifacts do "
                f"not provide measurements for {other_backends}. Neither artifact calibrates the "
                "synthetic scenarios or supports a named-product claim."
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
