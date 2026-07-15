from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalneurotwin.config import load_config
from causalneurotwin.run_contract import (
    RunContractError,
    default_run_id,
    execute_contract_demo,
    validate_run_id,
)


@pytest.fixture
def config_path() -> Path:
    return Path("configs/run_contract.example.yaml")


def test_explicit_run_id_produces_complete_bundle(config_path: Path, tmp_path: Path) -> None:
    run_dir = execute_contract_demo(
        config=load_config(config_path),
        config_path=config_path,
        output_root=tmp_path,
        run_id="deterministic-run",
        resume=False,
        command_line=["causalneurotwin", "run-contract", "--output-root", "/private/runs"],
    )
    expected = {
        "run_identity.json",
        "resolved_config.yaml",
        "environment.json",
        "input_manifest.json",
        "provenance.json",
        "events.jsonl",
        "run.log",
        "stdout.log",
        "metrics.json",
        "checksums.sha256",
        "RUN_COMPLETE",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    assert not (run_dir / "RUNNING").exists()
    assert not (run_dir / "RUN_FAILED").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["scientific_computation_performed"] is False
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert "/private/runs" not in str(provenance)


def test_default_ids_are_unique_and_prefixed() -> None:
    first = default_run_id("phase02")
    second = default_run_id("phase02")
    assert first.startswith("phase02-")
    assert first != second


def test_invalid_run_ids_are_rejected() -> None:
    with pytest.raises(RunContractError, match="run ID"):
        validate_run_id("../escape")


def test_existing_run_is_not_overwritten(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    execute_contract_demo(
        config=config,
        config_path=config_path,
        output_root=tmp_path,
        run_id="same",
        resume=False,
    )
    with pytest.raises(RunContractError, match="already exists"):
        execute_contract_demo(
            config=config,
            config_path=config_path,
            output_root=tmp_path,
            run_id="same",
            resume=False,
        )


def test_failure_marker_and_resume(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    with pytest.raises(RuntimeError, match="intentional"):
        execute_contract_demo(
            config=config,
            config_path=config_path,
            output_root=tmp_path,
            run_id="resumable",
            resume=False,
            fail_after_start=True,
        )
    failed = tmp_path / "resumable"
    assert (failed / "RUN_FAILED").exists()
    assert not (failed / "RUN_COMPLETE").exists()
    assert not (failed / "RUNNING").exists()
    assert (failed / "stderr.log").exists()

    resumed = execute_contract_demo(
        config=config,
        config_path=config_path,
        output_root=tmp_path,
        run_id="resumable",
        resume=True,
    )
    assert (resumed / "RUN_COMPLETE").exists()
    attempts = (resumed / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 1
    assert json.loads(attempts[0])["attempt"] == 2


def test_completed_run_cannot_resume(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    execute_contract_demo(
        config=config,
        config_path=config_path,
        output_root=tmp_path,
        run_id="complete",
        resume=False,
    )
    with pytest.raises(RunContractError, match="immutable"):
        execute_contract_demo(
            config=config,
            config_path=config_path,
            output_root=tmp_path,
            run_id="complete",
            resume=True,
        )


def test_required_manifest_is_hashed(config_path: Path, tmp_path: Path) -> None:
    payload = config_path.read_text(encoding="utf-8")
    payload = payload.replace("manifest_path: null", "manifest_path: pilot_manifest.json")
    payload = payload.replace("required: false", "required: true", 1)
    local_config = tmp_path / "config.yaml"
    local_manifest = tmp_path / "pilot_manifest.json"
    local_config.write_text(payload, encoding="utf-8")
    local_manifest.write_text('{"dataset": "pilot"}\n', encoding="utf-8")
    run_dir = execute_contract_demo(
        config=load_config(local_config),
        config_path=local_config,
        output_root=tmp_path / "runs",
        run_id="manifest",
        resume=False,
    )
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert [item["kind"] for item in manifest["inputs"]] == ["configuration", "data_manifest"]


def test_missing_required_manifest_creates_failure(config_path: Path, tmp_path: Path) -> None:
    payload = config_path.read_text(encoding="utf-8")
    payload = payload.replace("manifest_path: null", "manifest_path: missing.json")
    payload = payload.replace("required: false", "required: true", 1)
    local_config = tmp_path / "config.yaml"
    local_config.write_text(payload, encoding="utf-8")
    with pytest.raises(RunContractError, match="required data manifest"):
        execute_contract_demo(
            config=load_config(local_config),
            config_path=local_config,
            output_root=tmp_path / "runs",
            run_id="missing-manifest",
            resume=False,
        )
    assert (tmp_path / "runs" / "missing-manifest" / "RUN_FAILED").exists()


def test_resume_missing_run_is_rejected(config_path: Path, tmp_path: Path) -> None:
    with pytest.raises(RunContractError, match="missing run"):
        execute_contract_demo(
            config=load_config(config_path),
            config_path=config_path,
            output_root=tmp_path,
            run_id="missing",
            resume=True,
        )


def test_resume_requires_identity(config_path: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "no-identity"
    run_dir.mkdir()
    with pytest.raises(RunContractError, match="run_identity"):
        execute_contract_demo(
            config=load_config(config_path),
            config_path=config_path,
            output_root=tmp_path,
            run_id="no-identity",
            resume=True,
        )


def test_resume_rejects_changed_configuration(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    with pytest.raises(RuntimeError):
        execute_contract_demo(
            config=config,
            config_path=config_path,
            output_root=tmp_path,
            run_id="changed",
            resume=False,
            fail_after_start=True,
        )
    payload = config_path.read_text(encoding="utf-8").replace(
        "master_seed: 20260715", "master_seed: 9"
    )
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(payload, encoding="utf-8")
    with pytest.raises(RunContractError, match="does not match"):
        execute_contract_demo(
            config=load_config(changed_path),
            config_path=changed_path,
            output_root=tmp_path,
            run_id="changed",
            resume=True,
        )
