from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from memoryflow.io import load_request, write_json
from memoryflow.optimizer import pareto_front, sweep_hbm_windows
from memoryflow.simulator import simulate


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

    optimize_parser = subparsers.add_parser("optimize", help="sweep KV placement policies")
    optimize_parser.add_argument("scenario", type=Path)
    optimize_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = load_request(args.scenario)

    if args.command == "simulate":
        result = simulate(request).to_dict(include_steps=args.steps)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2))
        return 0

    results = sweep_hbm_windows(request)
    payload = {
        "all": [result.to_dict(include_steps=False) for result in results],
        "pareto_front": [result.to_dict(include_steps=False) for result in pareto_front(results)],
    }
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
