from __future__ import annotations

import subprocess
from pathlib import Path

from causalneurotwin.forbidden_data import is_forbidden, scan_repository


def test_known_participant_data_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "sub-001_T1w.nii.gz"
    path.write_bytes(b"not real data")
    assert is_forbidden(path) == "participant_data_model_or_secret_artifact"


def test_secret_configuration_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("TOKEN=secret", encoding="utf-8")
    assert is_forbidden(path) == "credential_or_private_configuration"


def test_zarr_directory_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "responses.zarr"
    path.mkdir()
    assert is_forbidden(path) == "scientific_array_store"


def test_safe_repository_has_no_findings(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("safe", encoding="utf-8")
    findings = scan_repository(tmp_path, tracked_only=False)
    assert findings == []


def test_scanner_finds_nested_artifact_and_ignores_build(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "raw"
    nested.mkdir(parents=True)
    (nested / "subject.edf").write_bytes(b"not real data")
    build = tmp_path / "build"
    build.mkdir()
    (build / "cached.pt").write_bytes(b"ignored build output")
    findings = scan_repository(tmp_path, tracked_only=False)
    assert [item.relative_path for item in findings] == ["data/raw/subject.edf"]


def test_scanner_uses_git_tracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    safe = tmp_path / "README.md"
    bad = tmp_path / "subject.fif"
    untracked_bad = tmp_path / "untracked.edf"
    safe.write_text("safe", encoding="utf-8")
    bad.write_bytes(b"not real data")
    untracked_bad.write_bytes(b"not real data")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "README.md", "subject.fif"],
        check=True,
    )
    findings = scan_repository(tmp_path, tracked_only=True)
    assert [item.relative_path for item in findings] == ["subject.fif"]
