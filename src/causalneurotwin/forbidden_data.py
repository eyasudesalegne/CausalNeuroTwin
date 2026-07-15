"""Repository scanner for participant data, credentials, and large artefacts."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".nii",
    ".nii.gz",
    ".mgz",
    ".mgh",
    ".edf",
    ".bdf",
    ".fif",
    ".set",
    ".vhdr",
    ".eeg",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".ckpt",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)
_FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        "credentials.json",
        "client_secret.json",
        "id_rsa",
        "id_ed25519",
    }
)
_IGNORED_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
)


@dataclass(frozen=True)
class Finding:
    """One prohibited repository item."""

    relative_path: str
    reason: str


def _normalised_suffix(path: Path) -> str:
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def is_forbidden(path: Path) -> str | None:
    """Return the reason a path is forbidden, or ``None`` when allowed."""

    name = path.name.lower()
    if name in _FORBIDDEN_EXACT_NAMES:
        return "credential_or_private_configuration"
    suffix = _normalised_suffix(path)
    if suffix in _FORBIDDEN_SUFFIXES:
        return "participant_data_model_or_secret_artifact"
    if name.endswith(".zarr") and path.is_dir():
        return "scientific_array_store"
    return None


def _git_tracked_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    entries = [item for item in result.stdout.decode("utf-8").split("\0") if item]
    return [root / item for item in entries]


def _walk_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in _IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_file() or (path.is_dir() and path.name.lower().endswith(".zarr")):
            yield path


def scan_repository(root: Path, *, tracked_only: bool = True) -> list[Finding]:
    """Scan tracked files when Git is available, otherwise scan the working tree."""

    root = root.resolve()
    candidates = _git_tracked_files(root) if tracked_only else None
    if candidates is None:
        candidates = list(_walk_files(root))

    findings: list[Finding] = []
    for path in candidates:
        reason = is_forbidden(path)
        if reason is not None:
            findings.append(Finding(relative_path=path.relative_to(root).as_posix(), reason=reason))
    return sorted(findings, key=lambda item: item.relative_path)
