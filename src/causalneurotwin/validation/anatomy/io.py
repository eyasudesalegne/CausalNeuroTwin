"""Public reusable evidence IO; subject-specific execution stays restricted."""

import csv
from pathlib import Path

from ..artifacts import clean, compute_guard
from ..artifacts import sha as digest
from ..artifacts import write as write_json

__all__ = [
    "CONSISTENCY",
    "INDEPENDENT",
    "PENDING",
    "clean",
    "compute_guard",
    "digest",
    "inside",
    "read_csv",
    "write_csv",
    "write_json",
]

CONSISTENCY = "computational_consistency_verified"
INDEPENDENT = "independent_anatomical_assessment_passed"
PENDING = "validation_pending_failed_or_indeterminate"


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(clean(rows))


def read_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h, delimiter=delimiter))


def inside(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False
