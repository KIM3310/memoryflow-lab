from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memoryflow.io import load_request, write_json
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    ROOT / "scenarios" / "7b-long-context-hbm-only.json",
    ROOT / "scenarios" / "7b-long-context-tiered.json",
    ROOT / "scenarios" / "7b-long-context-near-memory.json",
    ROOT / "scenarios" / "7b-long-context-near-memory-stress.json",
)


def _digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def build() -> dict[str, object]:
    results = [simulate(load_request(path)) for path in SCENARIOS]
    base_request = load_request(SCENARIOS[1])
    sweep = sweep_hbm_windows(base_request)
    frontier = pareto_front(sweep)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "scenario_sha256": _digest(SCENARIOS),
        "disclaimer": (
            "Synthetic first-order estimates for architectural comparison; "
            "not measured vendor-product performance."
        ),
        "results": [result.to_dict(include_steps=False) for result in results],
        "pareto_front": [result.to_dict(include_steps=False) for result in frontier],
    }
    write_json(ROOT / "site" / "results.json", payload)

    lines = [
        "# Reproducible Benchmark Summary",
        "",
        f"Scenario input SHA-256: `{payload['scenario_sha256']}`",
        "",
        (
            "These are synthetic first-order estimates for architecture comparison, "
            "not measured product claims."
        ),
        "",
        (
            "| Policy | Feasible | Mean decode (ms) | Throughput (token/s) "
            "| Remote read (GiB) | Bottleneck |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.policy_name,
                    "yes" if result.feasible else "no",
                    _format_number(result.mean_decode_latency_ms),
                    _format_number(result.throughput_tokens_s),
                    _format_number(result.total_remote_read_gib),
                    result.bottleneck,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- HBM-only is rejected because model weights plus full long-context KV "
                "exceed capacity."
            ),
            "- Naive tiering restores feasibility but exposes remote-transfer latency.",
            "- The near-memory proxy reduces transferred cold-KV bytes and wins in this scenario.",
            "- The slow-compute stress case reverses that win and exposes near-memory compute.",
            (
                "- The conclusion is conditional: change workload or bandwidth inputs "
                "and regenerate the evidence."
            ),
            "",
        ]
    )
    summary_path = ROOT / "evidence" / "benchmark-summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
