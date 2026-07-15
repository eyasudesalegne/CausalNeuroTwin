from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from causalneurotwin import cli


def test_doctor_human_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causalneurotwin",
            "doctor",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "CausalNeuroTwin 0.1.0a1" in result.stdout
    assert "Overall: ready" in result.stdout


def test_doctor_json_does_not_expose_data_root(tmp_path: Path) -> None:
    private_root = tmp_path / "private-subject-data"
    private_root.mkdir()
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causalneurotwin",
            "doctor",
            "--data-root",
            str(private_root),
            "--output-dir",
            str(output),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["data_root"]["status"] == "ready"
    assert str(private_root) not in result.stdout


def test_cli_main_human_output(tmp_path: Path, capsys: object) -> None:
    result = cli.main(["doctor", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert result == 0
    assert "Overall: ready" in captured.out


def test_cli_main_json_output(tmp_path: Path, capsys: object) -> None:
    result = cli.main(["doctor", "--output-dir", str(tmp_path), "--json"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert result == 0
    assert json.loads(captured.out)["overall_status"] == "ready"


def test_parser_has_doctor_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["doctor", "--json"])
    assert args.command == "doctor"
    assert args.json is True
