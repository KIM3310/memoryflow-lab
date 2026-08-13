from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Never, TypeVar, cast

from memoryflow.domain import (
    MemorySystem,
    PlacementPolicy,
    ScenarioProvenance,
    SimulationRequest,
    Workload,
)

SCENARIO_SCHEMA_VERSION = "2.0"
MAX_SCENARIO_BYTES = 1_048_576
T = TypeVar("T", Workload, MemorySystem, PlacementPolicy, ScenarioProvenance)


def _construct_strict(cls: type[T], payload: object, label: str) -> T:
    if not isinstance(payload, dict):
        raise ValueError(f"scenario {label} must be a JSON object")
    allowed = {field.name for field in fields(cls)}
    keys = set(payload)
    unknown = sorted(keys - allowed)
    missing = sorted(allowed - keys)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {label} fields: {', '.join(missing)}")
    try:
        return cls(**cast(dict[str, Any], payload))
    except TypeError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def request_from_dict(payload: object) -> SimulationRequest:
    if not isinstance(payload, dict):
        raise ValueError("scenario root must be a JSON object")
    expected_root = {"schema_version", "provenance", "workload", "system", "policy"}
    keys = set(payload)
    unknown = sorted(keys - expected_root)
    missing = sorted(expected_root - keys)
    if unknown:
        raise ValueError(f"unknown scenario fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing scenario fields: {', '.join(missing)}")
    if payload["schema_version"] != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            "unsupported scenario schema_version: "
            f"{payload['schema_version']!r}; expected {SCENARIO_SCHEMA_VERSION!r}"
        )
    request = SimulationRequest(
        workload=_construct_strict(Workload, payload["workload"], "workload"),
        system=_construct_strict(MemorySystem, payload["system"], "system"),
        policy=_construct_strict(PlacementPolicy, payload["policy"], "policy"),
        provenance=_construct_strict(ScenarioProvenance, payload["provenance"], "provenance"),
    )
    request.validate()
    return request


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        try:
            key.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "JSON strings and object keys must contain valid Unicode scalar values"
            ) from exc
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_standard_number(value: str) -> Never:
    raise ValueError(f"non-standard JSON number is not allowed: {value}")


def _validate_parsed_scalars(payload: object) -> None:
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "JSON strings and object keys must contain valid Unicode scalar values"
                ) from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def json_object_from_bytes(
    content: bytes, *, maximum_bytes: int = MAX_SCENARIO_BYTES
) -> dict[str, Any]:
    if len(content) > maximum_bytes:
        raise ValueError(f"JSON input exceeds {maximum_bytes:,} bytes")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON input must use UTF-8 encoding") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_standard_number,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON input is invalid: {exc.msg}") from exc
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the parser limit") from exc
    _validate_parsed_scalars(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return cast(dict[str, Any], payload)


def load_json_object(
    path: str | Path, *, maximum_bytes: int = MAX_SCENARIO_BYTES
) -> dict[str, Any]:
    return json_object_from_bytes(Path(path).read_bytes(), maximum_bytes=maximum_bytes)


def load_request(path: str | Path) -> SimulationRequest:
    return request_from_dict(load_json_object(path))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    destination.write_text(rendered, encoding="utf-8")
