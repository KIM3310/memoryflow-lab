from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from memoryflow.analysis import analyze_design_space
from memoryflow.io import load_request, write_json
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated numbers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoryflow",
        description="Reproducible LLM KV-cache memory co-design experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="run one scenario")
    simulate_parser.add_argument("scenario", type=Path)
    simulate_parser.add_argument("--output", type=Path)
    simulate_parser.add_argument("--steps", action="store_true")

    optimize_parser = subparsers.add_parser("optimize", help="sweep page-aligned HBM windows")
    optimize_parser.add_argument("scenario", type=Path)
    optimize_parser.add_argument("--output", type=Path)
    optimize_parser.add_argument(
        "--windows",
        type=_comma_ints,
        default=(128, 256, 512, 1024, 2048, 4096),
        help="comma-separated, page-aligned token counts",
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="run deterministic sensitivity and break-even analysis"
    )
    analyze_parser.add_argument("scenario", type=Path)
    analyze_parser.add_argument("--output", type=Path)
    analyze_parser.add_argument(
        "--link-bandwidths",
        type=_comma_floats,
        default=(32.0, 64.0, 128.0, 256.0),
    )
    analyze_parser.add_argument(
        "--near-memory-tops",
        type=_comma_floats,
        default=(0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 12.0),
    )
    analyze_parser.add_argument(
        "--sensitivity-multipliers",
        type=_comma_floats,
        default=(0.5, 0.8, 1.0, 1.2, 1.5),
    )
    return parser


def _emit(payload: dict[str, object], output: Path | None) -> None:
    if output:
        write_json(output, payload)
    else:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = load_request(args.scenario)

    if args.command == "simulate":
        _emit(simulate(request).to_dict(include_steps=args.steps), args.output)
        return 0

    if args.command == "optimize":
        results = sweep_hbm_windows(request, windows=args.windows)
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "all": [result.to_dict(include_steps=False) for result in results],
            "pareto_front": [
                result.to_dict(include_steps=False) for result in pareto_front(results)
            ],
        }
        _emit(payload, args.output)
        return 0

    report = analyze_design_space(
        request,
        remote_link_bandwidths_gbps=args.link_bandwidths,
        near_memory_tops_values=args.near_memory_tops,
        sensitivity_multipliers=args.sensitivity_multipliers,
    )
    _emit(report.to_dict(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
