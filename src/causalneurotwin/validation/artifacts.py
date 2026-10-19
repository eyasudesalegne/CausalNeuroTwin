"""Restricted runtime artifacts; never include generated bundles in Git."""

import json
import os
import socket
from pathlib import Path

import numpy as np

from causalneurotwin.checksums import sha256_file


def sha(path):
    return sha256_file(Path(path))


def safe(path, root=None):
    p = Path(path)
    if not p.is_absolute() or p.resolve() != p or any(q.is_symlink() for q in [p, *p.parents]):
        raise ValueError("Relative or redirected path")
    if root is not None and not p.is_relative_to(Path(root)):
        raise ValueError("Path outside project")
    return p


def compute_guard():
    host = socket.gethostname().split(".")[0]
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURMD_NODENAME", "").split(".")[0] != host
        or "login" in host
    ):
        raise RuntimeError("Allocated compute node required; no scientific login-node operation")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(clean(obj), f, indent=2, allow_nan=False)
        f.write("\n")


def save(path, **arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)
