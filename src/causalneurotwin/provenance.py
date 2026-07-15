"""Privacy-aware environment and source-control provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from causalneurotwin import __version__

_SAFE_SCHEDULER_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS_ON_NODE",
)
_TRACKED_PACKAGES = (
    "causalneurotwin",
    "PyYAML",
    "numpy",
    "torch",
    "nibabel",
    "mpi4py",
    "simnibs",
    "tvb-library",
)
_SECRET_ARGUMENTS = {
    "--token",
    "--password",
    "--secret",
    "--api-key",
    "--data-root",
    "--output-root",
}
_ABSOLUTE_WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")


def _run_git(repository_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def collect_git_provenance(repository_root: Path) -> dict[str, Any]:
    """Collect commit and dirty-state information without path disclosure."""

    commit = _run_git(repository_root, "rev-parse", "HEAD")
    status = _run_git(repository_root, "status", "--porcelain")
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _host_fingerprint() -> str:
    raw = socket.gethostname().encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * pages)


def collect_package_versions() -> dict[str, str | None]:
    """Collect a limited, reviewable dependency inventory."""

    versions: dict[str, str | None] = {}
    for distribution in _TRACKED_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    versions["causalneurotwin"] = __version__
    return versions


def _nvidia_smi() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []
    devices: list[dict[str, str]] = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {
                    "index": str(index),
                    "name": parts[0],
                    "memory_total_mib": parts[1],
                    "driver_version": parts[2],
                }
            )
    return devices


def redact_command_line(arguments: list[str]) -> list[str]:
    """Redact sensitive values and private absolute paths from a command line."""

    redacted: list[str] = []
    redact_next = False
    home = str(Path.home())
    for argument in arguments:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        key = argument.split("=", maxsplit=1)[0]
        if key in _SECRET_ARGUMENTS:
            if "=" in argument:
                redacted.append(f"{key}=<redacted>")
            else:
                redacted.append(argument)
                redact_next = True
            continue
        if (
            argument.startswith(home)
            or argument.startswith("/")
            or _ABSOLUTE_WINDOWS.match(argument)
        ):
            redacted.append(f"<path:{Path(argument).name or 'root'}>")
        else:
            redacted.append(argument)
    return redacted


def redact_text(text: str) -> str:
    """Redact common local path prefixes from diagnostic text."""

    replacements = {
        str(Path.home()): "<home>",
        str(Path.cwd()): "<working-directory>",
    }
    redacted = text
    for sensitive, replacement in replacements.items():
        if sensitive and sensitive not in {"/", "."}:
            redacted = redacted.replace(sensitive, replacement)
    return redacted


def collect_environment() -> dict[str, Any]:
    """Collect non-sensitive runtime and scheduler metadata."""

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "host_fingerprint": _host_fingerprint(),
        },
        "hardware": {
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _memory_bytes(),
            "nvidia_gpus": _nvidia_smi(),
        },
        "scheduler": {
            key: os.environ[key] for key in _SAFE_SCHEDULER_VARIABLES if key in os.environ
        },
        "packages": collect_package_versions(),
    }


def canonical_json_digest(payload: object) -> str:
    """Hash a JSON-compatible object deterministically."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
