"""Typed and validated configuration for CausalNeuroTwin runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

_CONFIG_SCHEMA_VERSION = "1.0"
T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration is incomplete, invalid, or ambiguous."""


def _require_mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    return cast(dict[str, Any], value)


def _check_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    missing = sorted(required - payload.keys())
    unknown = sorted(payload.keys() - required - optional)
    if missing:
        raise ConfigError(f"{location} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{location} contains unknown keys: {', '.join(unknown)}")


def _string(value: object, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{location} must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise ConfigError(f"{location} must not be empty")
    return result


def _integer(value: object, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{location} must be at least {minimum}")
    return value


def _number(value: object, location: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{location} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ConfigError(f"{location} must be positive")
    return result


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{location} must be a boolean")
    return value


def _optional_path(value: object, location: str) -> str | None:
    if value is None:
        return None
    result = _string(value, location)
    candidate = Path(result)
    if candidate.is_absolute():
        raise ConfigError(
            f"{location} must be relative or provided through an environment variable"
        )
    if ".." in candidate.parts:
        raise ConfigError(f"{location} must not escape its configured root")
    return candidate.as_posix()


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{location} must be a list of strings")
    result = tuple(_string(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ConfigError(f"{location} must not contain duplicates")
    return result


@dataclass(frozen=True)
class ProjectConfig:
    """Non-sensitive project metadata."""

    name: str
    phase: str
    description: str
    tags: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> ProjectConfig:
        payload = _require_mapping(value, "project")
        _check_keys(
            payload,
            required={"name", "phase", "description"},
            optional={"tags"},
            location="project",
        )
        tags = _string_tuple(payload.get("tags", []), "project.tags")
        return cls(
            name=_string(payload["name"], "project.name"),
            phase=_string(payload["phase"], "project.phase"),
            description=_string(payload["description"], "project.description"),
            tags=tags,
        )


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Independent random seeds and determinism policy."""

    master_seed: int
    dataset_split_seed: int
    simulation_seed: int
    model_seed: int
    dataloader_seed: int
    deterministic: bool

    @classmethod
    def from_mapping(cls, value: object) -> ReproducibilityConfig:
        payload = _require_mapping(value, "reproducibility")
        required = {
            "master_seed",
            "dataset_split_seed",
            "simulation_seed",
            "model_seed",
            "dataloader_seed",
            "deterministic",
        }
        _check_keys(payload, required=required, optional=set(), location="reproducibility")
        return cls(
            master_seed=_integer(payload["master_seed"], "reproducibility.master_seed", minimum=0),
            dataset_split_seed=_integer(
                payload["dataset_split_seed"], "reproducibility.dataset_split_seed", minimum=0
            ),
            simulation_seed=_integer(
                payload["simulation_seed"], "reproducibility.simulation_seed", minimum=0
            ),
            model_seed=_integer(payload["model_seed"], "reproducibility.model_seed", minimum=0),
            dataloader_seed=_integer(
                payload["dataloader_seed"], "reproducibility.dataloader_seed", minimum=0
            ),
            deterministic=_boolean(payload["deterministic"], "reproducibility.deterministic"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Dataset identity and optional non-sensitive manifest reference."""

    dataset_id: str
    dataset_version: str
    manifest_path: str | None
    required: bool

    @classmethod
    def from_mapping(cls, value: object) -> DataConfig:
        payload = _require_mapping(value, "data")
        _check_keys(
            payload,
            required={"dataset_id", "dataset_version", "required"},
            optional={"manifest_path"},
            location="data",
        )
        required = _boolean(payload["required"], "data.required")
        manifest = _optional_path(payload.get("manifest_path"), "data.manifest_path")
        if required and manifest is None:
            raise ConfigError("data.manifest_path is required when data.required is true")
        return cls(
            dataset_id=_string(payload["dataset_id"], "data.dataset_id"),
            dataset_version=_string(payload["dataset_version"], "data.dataset_version"),
            manifest_path=manifest,
            required=required,
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Reserved simulation contract with explicit SI time units."""

    enabled: bool
    engine: str
    duration_seconds: float | None
    time_step_seconds: float | None

    @classmethod
    def from_mapping(cls, value: object) -> SimulationConfig:
        payload = _require_mapping(value, "simulation")
        _check_keys(
            payload,
            required={"enabled", "engine"},
            optional={"duration_seconds", "time_step_seconds"},
            location="simulation",
        )
        enabled = _boolean(payload["enabled"], "simulation.enabled")
        duration_raw = payload.get("duration_seconds")
        step_raw = payload.get("time_step_seconds")
        duration = (
            None
            if duration_raw is None
            else _number(duration_raw, "simulation.duration_seconds", positive=True)
        )
        step = (
            None
            if step_raw is None
            else _number(step_raw, "simulation.time_step_seconds", positive=True)
        )
        if enabled and (duration is None or step is None):
            raise ConfigError(
                "simulation.duration_seconds and simulation.time_step_seconds are required "
                "when simulation.enabled is true"
            )
        if duration is not None and step is not None and step > duration:
            raise ConfigError("simulation.time_step_seconds must not exceed duration_seconds")
        return cls(
            enabled=enabled,
            engine=_string(payload["engine"], "simulation.engine"),
            duration_seconds=duration,
            time_step_seconds=step,
        )


@dataclass(frozen=True)
class ModelConfig:
    """Reserved model contract."""

    enabled: bool
    name: str
    precision: Literal["float64", "float32", "bfloat16", "float16"]

    @classmethod
    def from_mapping(cls, value: object) -> ModelConfig:
        payload = _require_mapping(value, "model")
        _check_keys(
            payload,
            required={"enabled", "name", "precision"},
            optional=set(),
            location="model",
        )
        precision = _string(payload["precision"], "model.precision")
        allowed = {"float64", "float32", "bfloat16", "float16"}
        if precision not in allowed:
            raise ConfigError(f"model.precision must be one of: {', '.join(sorted(allowed))}")
        return cls(
            enabled=_boolean(payload["enabled"], "model.enabled"),
            name=_string(payload["name"], "model.name"),
            precision=cast(Literal["float64", "float32", "bfloat16", "float16"], precision),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Reserved training contract."""

    enabled: bool
    epochs: int
    batch_size: int
    learning_rate: float | None

    @classmethod
    def from_mapping(cls, value: object) -> TrainingConfig:
        payload = _require_mapping(value, "training")
        _check_keys(
            payload,
            required={"enabled", "epochs", "batch_size"},
            optional={"learning_rate"},
            location="training",
        )
        enabled = _boolean(payload["enabled"], "training.enabled")
        epochs = _integer(payload["epochs"], "training.epochs", minimum=0)
        batch_size = _integer(payload["batch_size"], "training.batch_size", minimum=0)
        lr_raw = payload.get("learning_rate")
        learning_rate = (
            None if lr_raw is None else _number(lr_raw, "training.learning_rate", positive=True)
        )
        if enabled and (epochs < 1 or batch_size < 1 or learning_rate is None):
            raise ConfigError(
                "training.epochs, training.batch_size, and training.learning_rate must be "
                "positive when training.enabled is true"
            )
        return cls(
            enabled=enabled,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Reserved evaluation contract."""

    enabled: bool
    metrics: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> EvaluationConfig:
        payload = _require_mapping(value, "evaluation")
        _check_keys(
            payload,
            required={"enabled", "metrics"},
            optional=set(),
            location="evaluation",
        )
        enabled = _boolean(payload["enabled"], "evaluation.enabled")
        metrics = _string_tuple(payload["metrics"], "evaluation.metrics")
        if enabled and not metrics:
            raise ConfigError(
                "evaluation.metrics must not be empty when evaluation.enabled is true"
            )
        return cls(enabled=enabled, metrics=metrics)


@dataclass(frozen=True)
class HpcConfig:
    """Scheduler-independent resource layout."""

    scheduler: Literal["local", "slurm"]
    nodes: int
    tasks_per_node: int
    cpus_per_task: int
    gpus_per_node: int

    @classmethod
    def from_mapping(cls, value: object) -> HpcConfig:
        payload = _require_mapping(value, "hpc")
        _check_keys(
            payload,
            required={
                "scheduler",
                "nodes",
                "tasks_per_node",
                "cpus_per_task",
                "gpus_per_node",
            },
            optional=set(),
            location="hpc",
        )
        scheduler = _string(payload["scheduler"], "hpc.scheduler")
        if scheduler not in {"local", "slurm"}:
            raise ConfigError("hpc.scheduler must be 'local' or 'slurm'")
        return cls(
            scheduler=cast(Literal["local", "slurm"], scheduler),
            nodes=_integer(payload["nodes"], "hpc.nodes", minimum=1),
            tasks_per_node=_integer(payload["tasks_per_node"], "hpc.tasks_per_node", minimum=1),
            cpus_per_task=_integer(payload["cpus_per_task"], "hpc.cpus_per_task", minimum=1),
            gpus_per_node=_integer(payload["gpus_per_node"], "hpc.gpus_per_node", minimum=0),
        )


@dataclass(frozen=True)
class OutputConfig:
    """Retention and overwrite policy."""

    run_prefix: str
    retain_intermediate: bool
    checksum_algorithm: Literal["sha256"]
    overwrite: bool

    @classmethod
    def from_mapping(cls, value: object) -> OutputConfig:
        payload = _require_mapping(value, "output")
        _check_keys(
            payload,
            required={"run_prefix", "retain_intermediate", "checksum_algorithm", "overwrite"},
            optional=set(),
            location="output",
        )
        algorithm = _string(payload["checksum_algorithm"], "output.checksum_algorithm")
        if algorithm != "sha256":
            raise ConfigError("output.checksum_algorithm currently supports only 'sha256'")
        prefix = _string(payload["run_prefix"], "output.run_prefix")
        if not all(character.isalnum() or character in "-_" for character in prefix):
            raise ConfigError("output.run_prefix may contain only letters, digits, '-' and '_'")
        overwrite = _boolean(payload["overwrite"], "output.overwrite")
        if overwrite:
            raise ConfigError("output.overwrite must remain false for immutable Phase 02 runs")
        return cls(
            run_prefix=prefix,
            retain_intermediate=_boolean(
                payload["retain_intermediate"], "output.retain_intermediate"
            ),
            checksum_algorithm="sha256",
            overwrite=overwrite,
        )


@dataclass(frozen=True)
class RunConfig:
    """Complete Phase 02 configuration contract."""

    schema_version: str
    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    data: DataConfig
    simulation: SimulationConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    hpc: HpcConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, value: object) -> RunConfig:
        payload = _require_mapping(value, "configuration")
        required = {
            "schema_version",
            "project",
            "reproducibility",
            "data",
            "simulation",
            "model",
            "training",
            "evaluation",
            "hpc",
            "output",
        }
        _check_keys(payload, required=required, optional=set(), location="configuration")
        schema_version = _string(payload["schema_version"], "schema_version")
        if schema_version != _CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version must be {_CONFIG_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        return cls(
            schema_version=schema_version,
            project=ProjectConfig.from_mapping(payload["project"]),
            reproducibility=ReproducibilityConfig.from_mapping(payload["reproducibility"]),
            data=DataConfig.from_mapping(payload["data"]),
            simulation=SimulationConfig.from_mapping(payload["simulation"]),
            model=ModelConfig.from_mapping(payload["model"]),
            training=TrainingConfig.from_mapping(payload["training"]),
            evaluation=EvaluationConfig.from_mapping(payload["evaluation"]),
            hpc=HpcConfig.from_mapping(payload["hpc"]),
            output=OutputConfig.from_mapping(payload["output"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe representation."""

        return asdict(self)


def load_config(path: Path) -> RunConfig:
    """Load and validate a YAML run configuration."""

    if not path.exists():
        raise ConfigError(f"configuration file does not exist: {path.name}")
    if not path.is_file():
        raise ConfigError(f"configuration path is not a file: {path.name}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read configuration {path.name}: {exc}") from exc
    return RunConfig.from_mapping(loaded)
