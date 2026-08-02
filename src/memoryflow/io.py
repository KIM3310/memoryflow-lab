from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memoryflow.domain import MemorySystem, PlacementPolicy, SimulationRequest, Workload


def request_from_dict(payload: dict[str, Any]) -> SimulationRequest:
    return SimulationRequest(
        workload=Workload(**payload["workload"]),
        system=MemorySystem(**payload["system"]),
        policy=PlacementPolicy(**payload["policy"]),
    )


def load_request(path: str | Path) -> SimulationRequest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario root must be a JSON object")
    return request_from_dict(payload)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
