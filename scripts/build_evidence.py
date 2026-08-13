from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from memoryflow import __version__
from memoryflow.analysis import (
    ANALYSIS_SCHEMA_VERSION,
    BreakEvenPoint,
    Counterexample,
    DesignPoint,
    DesignSpaceReport,
    SensitivityEnvelope,
    SensitivityPoint,
    analyze_design_space,
)
from memoryflow.domain import RESULT_SCHEMA_VERSION, SimulationResult
from memoryflow.io import SCENARIO_SCHEMA_VERSION, load_json_object, load_request, write_json
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate
from scripts.build_measurement_summary import validate_committed_apple_m4_pair

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SCHEMA_VERSION = "2.0"
SCENARIOS = (
    ROOT / "scenarios" / "7b-long-context-hbm-only.json",
    ROOT / "scenarios" / "7b-long-context-tiered.json",
    ROOT / "scenarios" / "7b-long-context-near-memory.json",
    ROOT / "scenarios" / "7b-long-context-near-memory-stress.json",
)
MEASUREMENTS = (
    ROOT / "evidence" / "measurements" / "apple-m4-mps-copy.json",
    ROOT / "evidence" / "measurements" / "apple-m4-mps-attention.json",
)
MODEL_SOURCES = (
    ROOT / "src" / "memoryflow" / "domain.py",
    ROOT / "src" / "memoryflow" / "io.py",
    ROOT / "src" / "memoryflow" / "simulator.py",
    ROOT / "src" / "memoryflow" / "optimizer.py",
    ROOT / "src" / "memoryflow" / "analysis.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _set_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = _relative(path).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _expect_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        raise ValueError(f"{label} schema mismatch: unknown={unknown}, missing={missing}")


def _artifact_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    aggregation = payload.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError(f"measurement aggregation is missing: {path}")
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "schema_version": payload.get("schema_version"),
        "kind": payload.get("kind"),
        "device_backend": payload.get("environment", {}).get("device_backend"),
        "aggregation_level": aggregation.get("level"),
        "raw_iterations_included": aggregation.get("raw_iterations_included"),
    }


def validate_payload(payload: dict[str, Any], *, check_files: bool = True) -> None:
    _expect_keys(
        payload,
        {
            "schema_version",
            "generator",
            "inputs",
            "scenario_set_sha256",
            "disclaimer",
            "uncertainty",
            "results",
            "pareto_front",
            "analysis",
            "measurement",
            "attention_measurement",
        },
        "evidence",
    )
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema")
    generator = payload.get("generator")
    inputs = payload.get("inputs")
    if not isinstance(generator, dict) or not isinstance(inputs, dict):
        raise ValueError("evidence generator and inputs must be objects")
    _expect_keys(
        generator,
        {
            "package",
            "package_version",
            "command",
            "model_sources",
            "model_source_sha256",
            "generator_path",
            "generator_sha256",
        },
        "evidence generator",
    )
    _expect_keys(inputs, {"scenarios", "measurements"}, "evidence inputs")
    if (
        generator.get("package") != "memoryflow-lab"
        or generator.get("package_version") != __version__
    ):
        raise ValueError("evidence generator package provenance is stale")
    generator_path = ROOT / "scripts" / "build_evidence.py"
    if generator.get("generator_path") != _relative(generator_path) or generator.get(
        "generator_sha256"
    ) != _sha256(generator_path):
        raise ValueError("evidence generator script provenance is stale")
    model_sources = generator.get("model_sources")
    if not isinstance(model_sources, list) or len(model_sources) != len(MODEL_SOURCES):
        raise ValueError("evidence model source provenance is incomplete")
    for record, path in zip(model_sources, MODEL_SOURCES, strict=True):
        if not isinstance(record, dict):
            raise ValueError("model-source provenance record must be an object")
        _expect_keys(record, {"path", "sha256"}, "model-source provenance")
        if record.get("path") != _relative(path):
            raise ValueError(f"model-source provenance path does not match {path}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("model-source provenance SHA-256 must be a 64-character string")

    scenarios = inputs.get("scenarios")
    measurements = inputs.get("measurements")
    if not isinstance(scenarios, list) or not isinstance(measurements, list):
        raise ValueError("evidence input manifests must be lists")
    if len(scenarios) != len(SCENARIOS) or len(measurements) != len(MEASUREMENTS):
        raise ValueError("evidence input manifests have the wrong length")
    for record in scenarios:
        if not isinstance(record, dict):
            raise ValueError("scenario provenance record must be an object")
        _expect_keys(
            record,
            {"path", "sha256", "schema_version", "hardware_profile", "measurement_scope"},
            "scenario provenance",
        )
        if (
            record.get("schema_version") != SCENARIO_SCHEMA_VERSION
            or record.get("hardware_profile") != "synthetic"
            or record.get("measurement_scope") != "none"
        ):
            raise ValueError("bundled scenarios must declare unmeasured synthetic hardware")
    for record in measurements:
        if not isinstance(record, dict):
            raise ValueError("measurement provenance record must be an object")
        _expect_keys(
            record,
            {
                "path",
                "sha256",
                "schema_version",
                "kind",
                "device_backend",
                "aggregation_level",
                "raw_iterations_included",
            },
            "measurement provenance",
        )
        if (
            record.get("schema_version") != "2.0"
            or record.get("device_backend") != "mps"
            or record.get("aggregation_level") != "summary_statistics"
            or record.get("raw_iterations_included") is not False
        ):
            raise ValueError("committed measurement provenance boundary is invalid")

    results = payload.get("results")
    frontier = payload.get("pareto_front")
    analysis = payload.get("analysis")
    if not isinstance(results, list) or not results or not isinstance(frontier, list):
        raise ValueError("evidence results must be non-empty lists")
    analysis_keys = {"schema_version"} | {field.name for field in fields(DesignSpaceReport)}
    if not isinstance(analysis, dict):
        raise ValueError("evidence analysis must be an object")
    _expect_keys(analysis, analysis_keys, "analysis")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("evidence analysis schema is invalid")
    nested_analysis: tuple[tuple[str, type[Any]], ...] = (
        ("points", DesignPoint),
        ("break_even", BreakEvenPoint),
        ("sensitivity", SensitivityPoint),
    )
    for key, contract in nested_analysis:
        entries = analysis.get(key)
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"analysis {key} must be a list")
        expected_keys = {field.name for field in fields(contract)}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"analysis {key} entry must be an object")
            _expect_keys(entry, expected_keys, f"analysis {key} entry")
    for key, contract in (
        ("one_at_a_time_envelope", SensitivityEnvelope),
        ("counterexample", Counterexample),
    ):
        entry = analysis.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"analysis {key} must be an object")
        _expect_keys(entry, {field.name for field in fields(contract)}, f"analysis {key}")
    result_keys = {"schema_version"} | {field.name for field in fields(SimulationResult)}
    result_keys.remove("steps")
    for result in [*results, *frontier]:
        if not isinstance(result, dict):
            raise ValueError("evidence simulation result must be an object")
        _expect_keys(result, result_keys, "simulation result")
        if result.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("evidence contains an invalid simulation result")
        if result.get("input_hardware_profile") != "synthetic":
            raise ValueError("bundled results must retain synthetic provenance")

    copy_payload = payload.get("measurement")
    attention_payload = payload.get("attention_measurement")
    if not isinstance(copy_payload, dict) or not isinstance(attention_payload, dict):
        raise ValueError("embedded measurement summaries must be objects")
    validate_committed_apple_m4_pair(copy_payload, attention_payload)

    if check_files:
        if payload.get("scenario_set_sha256") != _set_digest(SCENARIOS):
            raise ValueError("scenario-set digest does not match source files")
        if generator.get("model_source_sha256") != _set_digest(MODEL_SOURCES):
            raise ValueError("model-source digest does not match source files")
        for record, path in zip(model_sources, MODEL_SOURCES, strict=True):
            if record.get("sha256") != _sha256(path):
                raise ValueError(f"model-source provenance does not match {path}")
        for record, path in zip(scenarios, SCENARIOS, strict=True):
            if record.get("path") != _relative(path) or record.get("sha256") != _sha256(path):
                raise ValueError(f"scenario provenance does not match {path}")
        for record, path in zip(measurements, MEASUREMENTS, strict=True):
            if record.get("path") != _relative(path) or record.get("sha256") != _sha256(path):
                raise ValueError(f"measurement provenance does not match {path}")
        if copy_payload != load_json_object(MEASUREMENTS[0]):
            raise ValueError("embedded copy measurement differs from its source artifact")
        if attention_payload != load_json_object(MEASUREMENTS[1]):
            raise ValueError("embedded attention measurement differs from its source artifact")
    json.dumps(payload, allow_nan=False)


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def _build_payload() -> dict[str, Any]:
    scenario_payloads = [load_json_object(path) for path in SCENARIOS]
    requests = [load_request(path) for path in SCENARIOS]
    results = [simulate(request) for request in requests]
    base_request = requests[1]
    sweep = sweep_hbm_windows(base_request)
    frontier = pareto_front(sweep)
    copy_measurement = load_json_object(MEASUREMENTS[0])
    attention_measurement = load_json_object(MEASUREMENTS[1])
    validate_committed_apple_m4_pair(copy_measurement, attention_measurement)

    scenario_records = []
    for path, source in zip(SCENARIOS, scenario_payloads, strict=True):
        provenance = source["provenance"]
        scenario_records.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "schema_version": source["schema_version"],
                "hardware_profile": provenance["hardware_profile"],
                "measurement_scope": provenance["measurement_scope"],
            }
        )
    model_records = [{"path": _relative(path), "sha256": _sha256(path)} for path in MODEL_SOURCES]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generator": {
            "package": "memoryflow-lab",
            "package_version": __version__,
            "command": "python -m scripts.build_evidence",
            "generator_path": "scripts/build_evidence.py",
            "generator_sha256": _sha256(ROOT / "scripts" / "build_evidence.py"),
            "model_sources": model_records,
            "model_source_sha256": _set_digest(MODEL_SOURCES),
        },
        "inputs": {
            "scenarios": scenario_records,
            "measurements": [
                _artifact_record(path, measurement)
                for path, measurement in zip(
                    MEASUREMENTS,
                    (copy_measurement, attention_measurement),
                    strict=True,
                )
            ],
        },
        "scenario_set_sha256": _set_digest(SCENARIOS),
        "disclaimer": (
            "Scenario results use synthetic design knobs and deterministic first-order equations; "
            "they are not measured vendor-product performance."
        ),
        "uncertainty": (
            "Sensitivity ranges are deterministic one-at-a-time perturbations, not statistical "
            "confidence intervals. Apple M4 MPS aggregates validate only equation shape."
        ),
        "results": [result.to_dict(include_steps=False) for result in results],
        "pareto_front": [result.to_dict(include_steps=False) for result in frontier],
        "analysis": analyze_design_space(base_request).to_dict(),
        "measurement": copy_measurement,
        "attention_measurement": attention_measurement,
    }


def _render_summary(payload: dict[str, Any]) -> str:
    results = payload["results"]
    analysis = payload["analysis"]
    lines = [
        "# Reproducible Benchmark Summary",
        "",
        f"Scenario-set SHA-256: `{payload['scenario_set_sha256']}`",
        f"Model-source SHA-256: `{payload['generator']['model_source_sha256']}`",
        f"Generator SHA-256: `{payload['generator']['generator_sha256']}`",
        "",
        (
            "All policy results below use bundled synthetic hardware knobs. They are "
            "deterministic first-order estimates, not measurements or product claims."
        ),
        "",
        (
            "| Policy | Feasible | Mean decode (ms) | Throughput (token/s) | "
            "Remote media read (GiB) | Link read (GiB) | Page read amp. | Bottleneck |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["policy_name"]),
                    "yes" if result["feasible"] else "no",
                    _format_number(float(result["mean_decode_latency_ms"])),
                    _format_number(float(result["throughput_tokens_s"])),
                    _format_number(float(result["total_remote_memory_read_gib"])),
                    _format_number(float(result["total_interconnect_read_gib"])),
                    f"{float(result['remote_memory_read_amplification']):.4f}×",
                    str(result["bottleneck"]),
                ]
            )
            + " |"
        )

    counterexample = analysis["counterexample"]
    envelope = analysis["one_at_a_time_envelope"]
    minimum_speedup = envelope["minimum_near_memory_speedup"]
    maximum_speedup = envelope["maximum_near_memory_speedup"]
    if minimum_speedup is None or maximum_speedup is None:
        sensitivity_line = "No feasible sensitivity points were available for a speedup envelope."
    else:
        sensitivity_line = (
            "Across the listed feasible one-at-a-time multipliers, near-memory speedup spans "
            f"{float(minimum_speedup):.3f}× to {float(maximum_speedup):.3f}×."
        )
    counterexample_tops = counterexample["counterexample_near_memory_tops"]
    if counterexample_tops is None:
        counterexample_line = f"Counterexample status: {counterexample['status']}. " + str(
            counterexample["conclusion"]
        )
    else:
        counterexample_line = (
            "The computed counterexample lowers synthetic near-memory peak throughput from "
            f"{float(counterexample['baseline_near_memory_tops']):.3f} TOPS to "
            f"{float(counterexample_tops):.6f} TOPS; the winner changes from "
            f"{counterexample['baseline_winner']} to {counterexample['counterexample_winner']}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- HBM-only is rejected after subtracting the explicit runtime reserve and "
                "page-allocating the full KV cache."
            ),
            (
                "- Sliding-window tiering is feasible, but page-rounded cold KV traverses both "
                "remote media and the bandwidth-limited interconnect."
            ),
            (
                "- Near-memory attention still scans the same remote pages; it sends queries and "
                "returns `(output, max, sum)` partial state instead of returning cold KV."
            ),
            (
                "- The base near-memory point wins, while the committed slow-compute stress point "
                "loses. This is a model counterexample, not a hardware observation."
            ),
            "",
            "## Deterministic sensitivity",
            "",
            sensitivity_line,
            counterexample_line,
            "",
            "| Peak link bandwidth (GB/s) | Near-memory break-even peak TOPS | Status |",
            "|---:|---:|---|",
        ]
    )
    for point in analysis["break_even"]:
        threshold = point["near_memory_break_even_tops"]
        rendered = "not reached" if threshold is None else f"{float(threshold):.6f}"
        lines.append(
            f"| {float(point['remote_link_bandwidth_gbps']):.0f} | {rendered} | {point['status']} |"
        )
    lines.extend(
        [
            "",
            "The envelope is not a confidence interval: it covers only declared deterministic "
            "perturbations. Correlated uncertainty, queueing, kernels, topology, and hardware "
            "behavior remain outside the model.",
            "",
            "## Measurement boundary",
            "",
            (
                "The embedded Apple M4 MPS artifacts contain median/p95 aggregate summaries, not "
                "raw iterations. They check a device-copy equation and one-layer SDPA prediction "
                "shape; they do not validate HBM, CXL, remote media, near-memory/PIM, CUDA, or "
                "end-to-end serving. No measurement value calibrates the synthetic scenarios."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    payload = _build_payload()
    validate_payload(payload)
    write_json(ROOT / "site" / "results.json", payload)
    summary_path = ROOT / "evidence" / "benchmark-summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_render_summary(payload), encoding="utf-8")
    return payload


def check() -> None:
    payload = _build_payload()
    validate_payload(payload)
    expected_json = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    results_path = ROOT / "site" / "results.json"
    summary_path = ROOT / "evidence" / "benchmark-summary.md"
    stale: list[Path] = []
    if not results_path.exists() or results_path.read_text(encoding="utf-8") != expected_json:
        stale.append(results_path)
    expected_summary = _render_summary(payload)
    if not summary_path.exists() or summary_path.read_text(encoding="utf-8") != expected_summary:
        stale.append(summary_path)
    if stale:
        joined = ", ".join(str(path) for path in stale)
        raise SystemExit(f"generated evidence is stale: {joined}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="generate deterministic MemoryFlow evidence")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        check()
    else:
        print(json.dumps(build(), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
