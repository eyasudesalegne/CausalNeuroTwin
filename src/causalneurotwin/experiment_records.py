"""Validate disclosure-reviewed reproducibility links without exporting private bundles."""

from __future__ import annotations

import re
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")


def validate_experiment_record(record: dict[str, Any]) -> dict[str, Any]:
    """A draft may be incomplete. A completed, publishable record must be fully linked."""
    required = {
        "schema_version",
        "record_id",
        "status",
        "git_commit",
        "git_dirty",
        "config_version",
        "config_sha256",
        "software_versions",
        "code_manifest_sha256",
        "input_integrity_manifest_sha256",
        "slurm_job_id",
        "result_manifest_sha256",
        "disclosure_review",
        "linkage",
    }
    if set(record) != required or record["schema_version"] != 1:
        raise ValueError("Exact versioned reproducibility record schema required")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,80}", record["record_id"]):
        raise ValueError("Nonidentifying record ID required")
    if record["status"] not in {"draft", "blocked", "completed"}:
        raise ValueError("Unknown experiment state")
    commit = record["git_commit"]
    if commit is not None and not _SHA1.fullmatch(commit):
        raise ValueError("Full Git commit SHA required")
    if record["git_dirty"] is not None and not isinstance(record["git_dirty"], bool):
        raise ValueError("Git dirty state")
    for key in (
        "config_sha256",
        "code_manifest_sha256",
        "input_integrity_manifest_sha256",
        "result_manifest_sha256",
    ):
        if record[key] is not None and not _SHA256.fullmatch(record[key]):
            raise ValueError("Invalid integrity digest")
    if not isinstance(record["software_versions"], dict) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in record["software_versions"].items()
    ):
        raise ValueError("Explicit software version mapping required")
    review = record["disclosure_review"]
    if not isinstance(review, dict) or set(review) != {"approved", "reviewer", "date", "scope"}:
        raise ValueError("Disclosure review fields required")
    if review["approved"] is not True:
        if (
            record["slurm_job_id"] is not None
            or record["input_integrity_manifest_sha256"] is not None
            or record["result_manifest_sha256"] is not None
        ):
            raise ValueError(
                "Internal scheduler/input/result metadata needs explicit disclosure review"
            )
    elif not all(
        isinstance(review[k], str) and review[k].strip() for k in ("reviewer", "date", "scope")
    ):
        raise ValueError("Completed disclosure attestation required")
    if record["linkage"] not in {
        "unlinked",
        "verified_at_execution",
        "retrospective_source_hash_only",
    }:
        raise ValueError("Unknown source linkage")
    if record["status"] == "completed":
        fields = (
            "git_commit",
            "config_version",
            "config_sha256",
            "code_manifest_sha256",
            "input_integrity_manifest_sha256",
            "slurm_job_id",
            "result_manifest_sha256",
        )
        if any(not record[k] for k in fields) or not record["software_versions"]:
            raise ValueError("Completed experiment needs code/config/input/result/scheduler links")
        if record["git_dirty"] is not False or record["linkage"] != "verified_at_execution":
            raise ValueError(
                "Do not attribute historical/uncommitted execution to a later Git commit"
            )
        if not str(record["slurm_job_id"]).isdigit():
            raise ValueError("Explicit numeric Slurm job identifier required")
    return record
