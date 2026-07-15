"""Atomic serialisation helpers for run artefacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: object) -> None:
    """Write deterministic pretty JSON atomically."""

    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write safe, stable YAML atomically."""

    _atomic_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def append_text(path: Path, content: str) -> None:
    """Append text and force it to disk for durable event and log records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
