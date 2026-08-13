from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

REQUIRED_WHEEL_SUFFIXES = {
    "memoryflow/__init__.py",
    "memoryflow/py.typed",
    "memoryflow/site/index.html",
    "memoryflow/site/app.js",
    "memoryflow/site/styles.css",
    "memoryflow/site/results.json",
}
REQUIRED_SDIST_SUFFIXES = {
    "README.md",
    "LICENSE",
    "src/memoryflow/simulator.py",
    "site/index.html",
    "scenarios/7b-long-context-tiered.json",
}


def _contains_suffix(names: set[str], suffix: str) -> bool:
    return any(name == suffix or name.endswith(f"/{suffix}") for name in names)


def validate_distributions(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    missing_wheel = sorted(
        suffix for suffix in REQUIRED_WHEEL_SUFFIXES if not _contains_suffix(wheel_names, suffix)
    )
    if missing_wheel:
        raise ValueError(f"wheel is missing runtime files: {missing_wheel}")
    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = set(archive.getnames())
    missing_sdist = sorted(
        suffix for suffix in REQUIRED_SDIST_SUFFIXES if not _contains_suffix(sdist_names, suffix)
    )
    if missing_sdist:
        raise ValueError(f"source distribution is missing files: {missing_sdist}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="validate MemoryFlow wheel and source archive")
    parser.add_argument("directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_distributions(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
