"""Streaming checksums and deterministic checksum manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(root: Path, paths: Iterable[Path], destination: Path) -> None:
    """Write a sorted GNU-compatible checksum manifest atomically."""

    entries: list[str] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path == destination:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    temporary.replace(destination)
