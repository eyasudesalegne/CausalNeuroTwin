"""Inspect staged Git blobs, including contents that differ from the working tree."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from causalneurotwin.forbidden_data import Finding, is_forbidden

_RESTRICTED = {
    "results",
    "validation_runs",
    "review_exports",
    ".test_artifacts",
    "credentials",
    "secrets",
    "private_configs",
    "private_records",
}
_BINARY = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".npy",
    ".npz",
    ".gii",
    ".msh",
    ".mat",
    ".pkl",
    ".pickle",
}
_PATTERNS = {
    "private_key_material": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "credential_token": re.compile(
        rb"(?:gh[pousr]_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,}|AKIA[A-Z0-9]{16})"
    ),
    "private_absolute_path": re.compile(
        rb"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|[/]valhalla[/]|[/]home[/][a-zA-Z0-9_.-]+[/])"
    ),
    "private_hpc_address": re.compile(rb"[a-zA-Z0-9_.-]+@login[.][a-zA-Z0-9_.-]+"),
}
MAX_BYTES = 1_000_000


def inspect_blob(relative_path: str, data: bytes, mode: str = "100644") -> list[Finding]:
    """Conservative disclosure gate; findings never print matching secret content."""
    path = Path(relative_path)
    reasons = []
    if mode not in {"100644", "100755"}:
        reasons.append("symlink_submodule_or_unmerged_entry")
    reason = is_forbidden(path)
    if reason:
        reasons.append(reason)
    if any(p.lower() in _RESTRICTED or "expert_review" in p.lower() for p in path.parts):
        reasons.append("restricted_output_or_review_directory")
    if path.suffix.lower() in _BINARY:
        reasons.append("scientific_array_image_or_archive_requires_disclosure_review")
    if len(data) > MAX_BYTES:
        reasons.append("oversized_source_requires_manual_review")
    if b"\0" in data:
        reasons.append("binary_content_not_approved")
    if len(data) <= MAX_BYTES:
        for name, pattern in _PATTERNS.items():
            if pattern.search(data):
                reasons.append(name)
    return [Finding(relative_path, x) for x in sorted(set(reasons))]


def scan_staged(root: Path) -> list[Finding]:
    """Scan the exact index objects, not unstaged replacements or ignored directories."""
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"], check=True, capture_output=True
    ).stdout
    records: list[tuple[str, str, str, str]] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header_raw, name_raw = entry.split(b"\t", 1)
        mode, oid, stage = header_raw.decode("ascii").split()
        records.append((mode, oid, name_raw.decode("utf-8"), stage))
    request = "".join(oid + "\n" for mode, oid, path, stage in records).encode("ascii")
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=request,
        check=True,
        capture_output=True,
    ).stdout
    offset = 0
    findings = []
    for mode, oid, path, stage in records:
        end = result.index(b"\n", offset)
        blob_header = result[offset:end].split()
        if (
            len(blob_header) != 3
            or blob_header[1] != b"blob"
            or blob_header[0].decode("ascii") != oid
        ):
            raise RuntimeError("Expected staged Git blob")
        size = int(blob_header[2])
        offset = end + 1
        data = result[offset : offset + size]
        offset += size + 1
        if stage != "0":
            mode = "unmerged"
        findings.extend(inspect_blob(path, data, mode))
    return sorted(findings, key=lambda x: (x.relative_path, x.reason))
