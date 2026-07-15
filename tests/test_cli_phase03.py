from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from causalneurotwin.cli import main


def create_fixture(root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/create_phase03_fixture.py", "--output", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dataset_validate_cli(tmp_path: Path, capsys: object) -> None:
    dataset_root = tmp_path / "dataset"
    create_fixture(dataset_root)
    output_dir = tmp_path / "report"
    exit_code = main(
        [
            "dataset",
            "validate",
            "--registry",
            "configs/data/openneuro_ds004024.yaml",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "pass"
    assert str(dataset_root) not in captured.out
    assert (output_dir / "dataset_validation.json").exists()


def test_dataset_validate_cli_missing_root(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "dataset",
            "validate",
            "--registry",
            "configs/data/openneuro_ds004024.yaml",
            "--dataset-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert "error:" in captured.err
    assert str(tmp_path) not in captured.err
