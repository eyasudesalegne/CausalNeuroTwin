from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from causalneurotwin.config import ConfigError, RunConfig, load_config


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "project": {
            "name": "CausalNeuroTwin",
            "phase": "phase-02",
            "description": "test configuration",
            "tags": ["test"],
        },
        "reproducibility": {
            "master_seed": 1,
            "dataset_split_seed": 2,
            "simulation_seed": 3,
            "model_seed": 4,
            "dataloader_seed": 5,
            "deterministic": True,
        },
        "data": {
            "dataset_id": "none",
            "dataset_version": "1",
            "manifest_path": None,
            "required": False,
        },
        "simulation": {
            "enabled": False,
            "engine": "not-implemented",
            "duration_seconds": None,
            "time_step_seconds": None,
        },
        "model": {"enabled": False, "name": "none", "precision": "float32"},
        "training": {
            "enabled": False,
            "epochs": 0,
            "batch_size": 0,
            "learning_rate": None,
        },
        "evaluation": {"enabled": False, "metrics": []},
        "hpc": {
            "scheduler": "local",
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": 1,
            "gpus_per_node": 0,
        },
        "output": {
            "run_prefix": "test",
            "retain_intermediate": False,
            "checksum_algorithm": "sha256",
            "overwrite": False,
        },
    }


def test_example_configuration_loads() -> None:
    config = load_config(Path("configs/run_contract.example.yaml"))
    assert config.schema_version == "1.0"
    assert config.simulation.enabled is False
    assert config.training.enabled is False
    assert config.to_dict()["project"]["name"] == "CausalNeuroTwin"


def test_unknown_key_is_rejected(valid_payload: dict[str, object]) -> None:
    valid_payload["unexpected"] = True
    with pytest.raises(ConfigError, match="unknown keys"):
        RunConfig.from_mapping(valid_payload)


def test_missing_section_is_rejected(valid_payload: dict[str, object]) -> None:
    del valid_payload["hpc"]
    with pytest.raises(ConfigError, match="missing required keys"):
        RunConfig.from_mapping(valid_payload)


def test_schema_version_is_pinned(valid_payload: dict[str, object]) -> None:
    valid_payload["schema_version"] = "9.0"
    with pytest.raises(ConfigError, match="schema_version"):
        RunConfig.from_mapping(valid_payload)


def test_enabled_simulation_requires_positive_explicit_times(
    valid_payload: dict[str, object],
) -> None:
    valid_payload["simulation"] = {
        "enabled": True,
        "engine": "reference",
        "duration_seconds": 1.0,
        "time_step_seconds": -0.01,
    }
    with pytest.raises(ConfigError, match="positive"):
        RunConfig.from_mapping(valid_payload)


def test_step_cannot_exceed_duration(valid_payload: dict[str, object]) -> None:
    valid_payload["simulation"] = {
        "enabled": True,
        "engine": "reference",
        "duration_seconds": 1.0,
        "time_step_seconds": 2.0,
    }
    with pytest.raises(ConfigError, match="must not exceed"):
        RunConfig.from_mapping(valid_payload)


def test_enabled_training_requires_complete_positive_values(
    valid_payload: dict[str, object],
) -> None:
    valid_payload["training"] = {
        "enabled": True,
        "epochs": 1,
        "batch_size": 0,
        "learning_rate": 0.001,
    }
    with pytest.raises(ConfigError, match="must be positive"):
        RunConfig.from_mapping(valid_payload)


def test_absolute_manifest_path_is_rejected(valid_payload: dict[str, object]) -> None:
    valid_payload["data"] = {
        "dataset_id": "x",
        "dataset_version": "1",
        "manifest_path": "/private/data/manifest.json",
        "required": True,
    }
    with pytest.raises(ConfigError, match="must be relative"):
        RunConfig.from_mapping(valid_payload)


def test_duplicate_metrics_are_rejected(valid_payload: dict[str, object]) -> None:
    valid_payload["evaluation"] = {"enabled": True, "metrics": ["mse", "mse"]}
    with pytest.raises(ConfigError, match="duplicates"):
        RunConfig.from_mapping(valid_payload)


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("x: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not read"):
        load_config(path)


def test_nonexistent_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_yaml_round_trip_fixture(valid_payload: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(valid_payload), encoding="utf-8")
    assert load_config(path).project.phase == "phase-02"


def test_required_manifest_must_be_declared(valid_payload: dict[str, object]) -> None:
    valid_payload["data"] = {
        "dataset_id": "x",
        "dataset_version": "1",
        "required": True,
    }
    with pytest.raises(ConfigError, match="manifest_path is required"):
        RunConfig.from_mapping(valid_payload)


def test_enabled_sections_can_validate(valid_payload: dict[str, object]) -> None:
    valid_payload["simulation"] = {
        "enabled": True,
        "engine": "reference",
        "duration_seconds": 1.0,
        "time_step_seconds": 0.01,
    }
    valid_payload["training"] = {
        "enabled": True,
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
    }
    valid_payload["evaluation"] = {"enabled": True, "metrics": ["mse"]}
    config = RunConfig.from_mapping(valid_payload)
    assert config.simulation.time_step_seconds == 0.01
    assert config.training.learning_rate == 0.001
    assert config.evaluation.metrics == ("mse",)


def test_invalid_model_precision_is_rejected(valid_payload: dict[str, object]) -> None:
    valid_payload["model"] = {"enabled": True, "name": "x", "precision": "int8"}
    with pytest.raises(ConfigError, match=r"model\.precision"):
        RunConfig.from_mapping(valid_payload)


def test_invalid_scheduler_is_rejected(valid_payload: dict[str, object]) -> None:
    hpc = dict(valid_payload["hpc"])  # type: ignore[arg-type]
    hpc["scheduler"] = "pbs"
    valid_payload["hpc"] = hpc
    with pytest.raises(ConfigError, match="scheduler"):
        RunConfig.from_mapping(valid_payload)


def test_overwrite_is_rejected(valid_payload: dict[str, object]) -> None:
    output = dict(valid_payload["output"])  # type: ignore[arg-type]
    output["overwrite"] = True
    valid_payload["output"] = output
    with pytest.raises(ConfigError, match="must remain false"):
        RunConfig.from_mapping(valid_payload)


def test_invalid_run_prefix_is_rejected(valid_payload: dict[str, object]) -> None:
    output = dict(valid_payload["output"])  # type: ignore[arg-type]
    output["run_prefix"] = "../bad"
    valid_payload["output"] = output
    with pytest.raises(ConfigError, match="run_prefix"):
        RunConfig.from_mapping(valid_payload)


def test_configuration_root_must_be_mapping() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        RunConfig.from_mapping([])


def test_configuration_path_must_be_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a file"):
        load_config(tmp_path)
