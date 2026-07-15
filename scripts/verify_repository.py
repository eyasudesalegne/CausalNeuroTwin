#!/usr/bin/env python3
"""Fail when prohibited data or secret artefacts are present in the repository."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causalneurotwin.forbidden_data import scan_repository  # noqa: E402


def main() -> int:
    findings = scan_repository(REPOSITORY_ROOT, tracked_only=True)
    if findings:
        print("Repository verification failed. Prohibited files were detected:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding.relative_path}: {finding.reason}", file=sys.stderr)
        return 1
    print("Repository verification passed: no prohibited tracked artefacts detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
