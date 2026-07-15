from __future__ import annotations

from pathlib import Path

from causalneurotwin.cli import main


def test_run_contract_cli_success(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "run-contract",
            "--config",
            "configs/run_contract.example.yaml",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "cli-test",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert "cli-test" in captured.out
    assert (tmp_path / "cli-test" / "RUN_COMPLETE").exists()


def test_run_contract_cli_reports_config_error(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: '1.0'\n", encoding="utf-8")
    exit_code = main(["run-contract", "--config", str(path), "--output-root", str(tmp_path)])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert "error:" in captured.err


def test_run_contract_cli_failure_path(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "run-contract",
            "--config",
            "configs/run_contract.example.yaml",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "cli-failure",
            "--exercise-failure",
        ]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert "intentional" in captured.err
    assert (tmp_path / "cli-failure" / "RUN_FAILED").exists()
