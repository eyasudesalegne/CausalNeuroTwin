"""Immutable, restart-aware Phase 02 run directories."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from causalneurotwin.checksums import sha256_file, write_checksum_manifest
from causalneurotwin.config import RunConfig
from causalneurotwin.provenance import (
    canonical_json_digest,
    collect_environment,
    collect_git_provenance,
    redact_command_line,
    redact_text,
)
from causalneurotwin.serialization import append_text, write_json, write_yaml

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class RunContractError(RuntimeError):
    """Raised when a run directory cannot be created or resumed safely."""


def utc_now() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def default_run_id(prefix: str) -> str:
    """Create a sortable, collision-resistant run identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    """Validate a user-supplied identifier before using it as a directory name."""

    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise RunContractError(
            "run ID must be 1-96 characters and contain only letters, digits, '.', '_' or '-'"
        )
    return run_id


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class RunSession:
    """Lifecycle manager for one immutable or safely resumed run."""

    config: RunConfig
    config_path: Path
    output_root: Path
    run_id: str
    resume: bool = False
    command_line: list[str] = field(default_factory=lambda: list(sys.argv))
    run_dir: Path = field(init=False)
    started_at: str = field(init=False)
    started_monotonic: float = field(init=False)
    config_digest: str = field(init=False)
    attempt: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.run_id = validate_run_id(self.run_id)
        self.run_dir = self.output_root / self.run_id
        self.config_digest = canonical_json_digest(self.config.to_dict())

    @property
    def _identity_path(self) -> Path:
        return self.run_dir / "run_identity.json"

    def _event(self, event: str, **details: object) -> None:
        payload = {
            "timestamp_utc": utc_now(),
            "event": event,
            "attempt": self.attempt,
            **details,
        }
        append_text(self.run_dir / "events.jsonl", json.dumps(payload, sort_keys=True) + "\n")
        rendered = f"{payload['timestamp_utc']} [{event}]"
        if details:
            rendered += " " + json.dumps(details, sort_keys=True)
        append_text(self.run_dir / "run.log", rendered + "\n")

    def _prepare_new(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            self.run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise RunContractError(
                f"run directory already exists for {self.run_id!r}; use --resume only for an "
                "incomplete run with matching configuration"
            ) from exc
        write_json(
            self._identity_path,
            {
                "run_id": self.run_id,
                "configuration_digest_sha256": self.config_digest,
                "schema_version": self.config.schema_version,
                "created_at_utc": utc_now(),
            },
        )

    def _prepare_resume(self) -> None:
        if not self.run_dir.is_dir():
            raise RunContractError(f"cannot resume missing run {self.run_id!r}")
        if (self.run_dir / "RUN_COMPLETE").exists():
            raise RunContractError("completed runs are immutable and cannot be resumed")
        if not self._identity_path.is_file():
            raise RunContractError("cannot resume a run without run_identity.json")
        try:
            identity = json.loads(self._identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunContractError("cannot resume because run_identity.json is unreadable") from exc
        if identity.get("configuration_digest_sha256") != self.config_digest:
            raise RunContractError("resume configuration does not match the original run")
        attempts_path = self.run_dir / "attempts.jsonl"
        if attempts_path.exists():
            self.attempt = 2 + sum(1 for _ in attempts_path.open(encoding="utf-8"))
        else:
            self.attempt = 2
        append_text(
            attempts_path,
            json.dumps({"attempt": self.attempt, "resumed_at_utc": utc_now()}, sort_keys=True)
            + "\n",
        )
        (self.run_dir / "RUNNING").unlink(missing_ok=True)
        (self.run_dir / "RUN_FAILED").unlink(missing_ok=True)

    def start(self) -> RunSession:
        """Create or safely reopen the run and write immutable startup records."""

        if self.resume:
            self._prepare_resume()
        else:
            self._prepare_new()
        self.started_at = utc_now()
        self.started_monotonic = time.monotonic()
        write_json(
            self.run_dir / "RUNNING",
            {"started_at_utc": self.started_at, "attempt": self.attempt, "pid": os.getpid()},
        )
        write_yaml(self.run_dir / "resolved_config.yaml", self.config.to_dict())
        environment = collect_environment()
        write_json(self.run_dir / "environment.json", environment)
        config_input = {
            "kind": "configuration",
            "name": self.config_path.name,
            "size_bytes": self.config_path.stat().st_size,
            "sha256": sha256_file(self.config_path),
        }
        manifest_inputs: list[dict[str, object]] = [config_input]
        if self.config.data.manifest_path is not None:
            manifest_path = self.config_path.parent / self.config.data.manifest_path
            if manifest_path.exists() and manifest_path.is_file():
                manifest_inputs.append(
                    {
                        "kind": "data_manifest",
                        "name": manifest_path.name,
                        "size_bytes": manifest_path.stat().st_size,
                        "sha256": sha256_file(manifest_path),
                    }
                )
            elif self.config.data.required:
                raise RunContractError(
                    "required data manifest is unavailable: "
                    f"{Path(self.config.data.manifest_path).name}"
                )
        write_json(
            self.run_dir / "input_manifest.json",
            {
                "dataset_id": self.config.data.dataset_id,
                "dataset_version": self.config.data.dataset_version,
                "inputs": manifest_inputs,
            },
        )
        write_json(
            self.run_dir / "provenance.json",
            {
                "run_id": self.run_id,
                "attempt": self.attempt,
                "started_at_utc": self.started_at,
                "configuration_digest_sha256": self.config_digest,
                "command_line": redact_command_line(self.command_line),
                "git": collect_git_provenance(_repository_root()),
                "seeds": {
                    "master": self.config.reproducibility.master_seed,
                    "dataset_split": self.config.reproducibility.dataset_split_seed,
                    "simulation": self.config.reproducibility.simulation_seed,
                    "model": self.config.reproducibility.model_seed,
                    "dataloader": self.config.reproducibility.dataloader_seed,
                },
                "deterministic_requested": self.config.reproducibility.deterministic,
            },
        )
        self._event("run_started", run_id=self.run_id)
        return self

    def write_stdout(self, message: str) -> None:
        """Append a durable stdout-equivalent record."""

        append_text(self.run_dir / "stdout.log", message.rstrip() + "\n")

    def complete(self, metrics: dict[str, object]) -> None:
        """Validate final artefacts, checksum them, and atomically mark completion."""

        elapsed = time.monotonic() - self.started_monotonic
        completed_at = utc_now()
        final_metrics = {
            **metrics,
            "status": "success",
            "run_id": self.run_id,
            "attempt": self.attempt,
            "elapsed_seconds": round(elapsed, 6),
            "completed_at_utc": completed_at,
        }
        write_json(self.run_dir / "metrics.json", final_metrics)
        self._event("run_completed", elapsed_seconds=round(elapsed, 6))
        checksum_targets = [
            path
            for path in self.run_dir.iterdir()
            if path.is_file()
            and path.name not in {"RUNNING", "RUN_COMPLETE", "RUN_FAILED", "checksums.sha256"}
        ]
        write_checksum_manifest(self.run_dir, checksum_targets, self.run_dir / "checksums.sha256")
        write_json(
            self.run_dir / "RUN_COMPLETE",
            {"completed_at_utc": completed_at, "attempt": self.attempt},
        )
        (self.run_dir / "RUNNING").unlink(missing_ok=True)
        (self.run_dir / "RUN_FAILED").unlink(missing_ok=True)

    def fail(self, exc: BaseException) -> None:
        """Record a structured failure without claiming completion."""

        failed_at = utc_now()
        error = {
            "status": "failed",
            "run_id": self.run_id,
            "attempt": self.attempt,
            "failed_at_utc": failed_at,
            "exception_type": type(exc).__name__,
            "message": redact_text(str(exc)),
        }
        write_json(self.run_dir / "failure.json", error)
        append_text(
            self.run_dir / "stderr.log",
            redact_text("".join(traceback.format_exception(exc))),
        )
        self._event("run_failed", exception_type=type(exc).__name__)
        write_json(self.run_dir / "RUN_FAILED", error)
        (self.run_dir / "RUNNING").unlink(missing_ok=True)


def execute_contract_demo(
    *,
    config: RunConfig,
    config_path: Path,
    output_root: Path,
    run_id: str | None,
    resume: bool,
    command_line: list[str] | None = None,
    fail_after_start: bool = False,
) -> Path:
    """Exercise the run contract without claiming a scientific computation."""

    resolved_run_id = run_id or default_run_id(config.output.run_prefix)
    session = RunSession(
        config=config,
        config_path=config_path,
        output_root=output_root,
        run_id=resolved_run_id,
        resume=resume,
        command_line=list(sys.argv) if command_line is None else command_line,
    )
    try:
        session.start()
        if fail_after_start:
            raise RuntimeError("intentional Phase 02 failure-path exercise")
        session.write_stdout(
            "Phase 02 run-contract validation completed; no scientific simulation or training ran."
        )
        session.complete(
            {
                "contract_validation": True,
                "scientific_computation_performed": False,
                "configured_sections": 9,
            }
        )
    except BaseException as exc:
        if session.run_dir.exists() and hasattr(session, "started_at"):
            session.fail(exc)
        raise
    return session.run_dir
