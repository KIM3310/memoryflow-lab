from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from memoryflow import __version__
from memoryflow.analysis import analyze_design_space
from memoryflow.io import MAX_SCENARIO_BYTES, json_object_from_bytes, request_from_dict
from memoryflow.simulator import simulate

ROOT = Path(__file__).resolve().parents[2]


def resolve_site_directory() -> Path:
    configured = os.environ.get("MEMORYFLOW_SITE_DIR")
    bundled = Path(__file__).with_name("site")
    default_site = bundled if bundled.is_dir() else ROOT / "site"
    site = Path(configured).expanduser() if configured else default_site
    resolved = site.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"MemoryFlow dashboard directory does not exist: {resolved}")
    return resolved


SITE = resolve_site_directory()

app = FastAPI(
    title="MemoryFlow Lab",
    version=__version__,
    description="Deterministic LLM KV-cache memory co-design simulator",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "analytical-first-order"}


SCENARIO_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "description": (
                        "Exact MemoryFlow scenario schema 2.0; duplicate keys and bodies over "
                        f"{MAX_SCENARIO_BYTES:,} bytes are rejected before JSON parsing."
                    ),
                }
            }
        },
    }
}


async def _strict_payload(raw_request: Request) -> dict[str, Any]:
    media_type = raw_request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not (
        media_type.startswith("application/") and media_type.endswith("+json")
    ):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    content_length = raw_request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be a non-negative integer") from exc
        if declared_bytes < 0:
            raise ValueError("Content-Length must be a non-negative integer")
        if declared_bytes > MAX_SCENARIO_BYTES:
            raise ValueError(f"JSON input exceeds {MAX_SCENARIO_BYTES:,} bytes")

    content = bytearray()
    async for chunk in raw_request.stream():
        if len(content) + len(chunk) > MAX_SCENARIO_BYTES:
            raise ValueError(f"JSON input exceeds {MAX_SCENARIO_BYTES:,} bytes")
        content.extend(chunk)
    return json_object_from_bytes(bytes(content))


@app.post("/v1/simulations", openapi_extra=SCENARIO_REQUEST_BODY)
async def create_simulation(raw_request: Request) -> dict[str, Any]:
    try:
        request = request_from_dict(await _strict_payload(raw_request))
        return await run_in_threadpool(lambda: simulate(request).to_dict(include_steps=True))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/analyses", openapi_extra=SCENARIO_REQUEST_BODY)
async def create_analysis(raw_request: Request) -> dict[str, Any]:
    try:
        request = request_from_dict(await _strict_payload(raw_request))
        return await run_in_threadpool(lambda: analyze_design_space(request).to_dict())
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=SITE, html=True), name="site")
