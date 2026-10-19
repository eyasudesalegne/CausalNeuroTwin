import subprocess
from pathlib import Path

import pytest

from causalneurotwin.publication import inspect_blob, scan_staged


@pytest.mark.parametrize(
    "name",
    [
        "results/report.json",
        "validation_runs/report.csv",
        "review_exports/index.html",
        ".test_artifacts/x.txt",
        "evidence.npy",
        "figure.png",
        "package.zip",
    ],
)
def test_restricted_assets(name: str) -> None:
    assert inspect_blob(name, b"synthetic placeholder")


@pytest.mark.parametrize(
    "content",
    [
        "C" + ":" + chr(92) + "private" + chr(92) + "fixture",
        "/val" + "halla/projects/private",
        "example@" + "login.example.invalid",
        "gh" + "p_" + "a" * 30,
        "-----BEGIN " + "PRIVATE KEY-----",
    ],
)
def test_content_findings_without_disclosing_value(content: str) -> None:
    findings = inspect_blob("example.txt", content.encode())
    assert findings
    assert all(content not in x.reason for x in findings)


def test_safe_source_and_explicit_size_guard() -> None:
    assert not inspect_blob("src/module.py", b'"""safe algorithm"""')
    assert inspect_blob("src/module.py", b"x" * 1_000_001)


def test_staged_blob_is_scanned_even_if_working_tree_is_cleaned(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "notes.txt"
    p.write_text("gh" + "p_" + "a" * 30)
    subprocess.run(["git", "-C", str(tmp_path), "add", "notes.txt"], check=True)
    p.write_text("apparently safe unstaged replacement")
    assert any(x.reason == "credential_token" for x in scan_staged(tmp_path))


def test_staged_safe_file_does_not_scan_untracked_private_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "notes.txt").write_text("safe")
    (tmp_path / "secret.txt").write_text("gh" + "p_" + "a" * 30)
    subprocess.run(["git", "-C", str(tmp_path), "add", "notes.txt"], check=True)
    assert scan_staged(tmp_path) == []
