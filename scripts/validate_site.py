from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from memoryflow.io import load_json_object
from scripts.build_evidence import ROOT, validate_payload

REQUIRED_IDS = {
    "metric-feasible",
    "metric-reason",
    "metric-latency",
    "metric-throughput",
    "metric-media",
    "metric-link",
    "policy-selector",
    "comparison-chart",
    "break-even-value",
    "sensitivity-range",
    "counterexample-value",
    "break-even-table",
    "uncertainty-boundary",
    "results-table",
    "provenance-list",
    "disclaimer",
    "scenario-hash",
    "attention-roofline-mape",
    "attention-affine-mape",
    "attention-max-error",
    "attention-effective-rate",
    "attention-table",
    "attention-boundary",
    "copy-baseline-mape",
    "copy-affine-mape",
    "copy-bandwidth",
    "copy-base-latency",
    "copy-table",
    "copy-boundary",
}


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.local_assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id is not None:
            self.ids.append(element_id)
        asset = (
            values.get("src") if tag == "script" else values.get("href") if tag == "link" else None
        )
        if asset and not asset.startswith(("data:", "http://", "https://", "//")):
            self.local_assets.append(asset)


def validate_site(site: Path = ROOT / "site") -> None:
    index = site / "index.html"
    app = site / "app.js"
    results_path = site / "results.json"
    parser = SiteParser()
    parser.feed(index.read_text(encoding="utf-8"))
    duplicates = sorted({value for value in parser.ids if parser.ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"site contains duplicate element ids: {duplicates}")
    missing = sorted(REQUIRED_IDS - set(parser.ids))
    if missing:
        raise ValueError(f"site is missing required element ids: {missing}")
    for relative in parser.local_assets:
        candidate = (site / relative).resolve()
        if site.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"site asset is missing or escapes site root: {relative}")

    javascript = app.read_text(encoding="utf-8")
    for element_id in REQUIRED_IDS:
        if f"#{element_id}" not in javascript:
            raise ValueError(f"site JavaScript does not reference required element: {element_id}")
    for stale_field in ("total_remote_read_gib", "near_memory_reduction_ratio", "metric-traffic"):
        if stale_field in javascript or stale_field in index.read_text(encoding="utf-8"):
            raise ValueError(f"site references removed model field: {stale_field}")

    payload: dict[str, Any] = load_json_object(results_path, maximum_bytes=5_000_000)
    validate_payload(payload)
    if len(payload["results"]) != 4:
        raise ValueError("site must present all four bundled scenario results")


if __name__ == "__main__":
    validate_site()
    print("site validation passed")
