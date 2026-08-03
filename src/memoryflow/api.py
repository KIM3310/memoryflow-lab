from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from memoryflow.io import request_from_dict
from memoryflow.simulator import simulate

ROOT = Path(__file__).resolve().parents[2]


def resolve_site_directory() -> Path:
    configured = os.environ.get("MEMORYFLOW_SITE_DIR")
    site = Path(configured).expanduser() if configured else ROOT / "site"
    resolved = site.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"MemoryFlow dashboard directory does not exist: {resolved}")
    return resolved


SITE = resolve_site_directory()

app = FastAPI(
    title="MemoryFlow Lab",
    version="0.1.0",
    description="Deterministic LLM KV-cache memory co-design simulator",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "analytical-first-order"}


@app.post("/v1/simulations")
def create_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        request = request_from_dict(payload)
        return simulate(request).to_dict(include_steps=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=SITE, html=True), name="site")
