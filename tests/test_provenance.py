from __future__ import annotations

from pathlib import Path

from causalneurotwin.provenance import (
    canonical_json_digest,
    collect_environment,
    collect_git_provenance,
    redact_command_line,
)


def test_command_line_redacts_secrets_and_paths(tmp_path: Path) -> None:
    command = [
        "causalneurotwin",
        "run-contract",
        "--token",
        "secret",
        "--data-root=/private/subject-data",
        str(tmp_path / "config.yaml"),
        "safe-value",
    ]
    redacted = redact_command_line(command)
    assert "secret" not in redacted
    assert "/private/subject-data" not in " ".join(redacted)
    assert str(tmp_path) not in " ".join(redacted)
    assert "safe-value" in redacted


def test_environment_uses_hashed_host_and_allowlisted_scheduler(monkeypatch: object) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")  # type: ignore[attr-defined]
    monkeypatch.setenv("USER", "private-user")  # type: ignore[attr-defined]
    environment = collect_environment()
    platform_payload = environment["platform"]
    assert len(platform_payload["host_fingerprint"]) == 16
    assert environment["scheduler"] == {"SLURM_JOB_ID": "123"}
    assert "private-user" not in str(environment)


def test_digest_is_key_order_independent() -> None:
    assert canonical_json_digest({"a": 1, "b": 2}) == canonical_json_digest({"b": 2, "a": 1})


def test_git_provenance_gracefully_handles_nonrepository(tmp_path: Path) -> None:
    result = collect_git_provenance(tmp_path)
    assert result == {"available": False, "commit": None, "dirty": None}


def test_nvidia_smi_inventory(monkeypatch: object) -> None:
    import subprocess

    from causalneurotwin import provenance

    monkeypatch.setattr(provenance.shutil, "which", lambda _: "/usr/bin/nvidia-smi")  # type: ignore[attr-defined]
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="NVIDIA H200, 141000, 600.00\n",
        stderr="",
    )
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: completed)  # type: ignore[attr-defined]
    environment = provenance.collect_environment()
    assert environment["hardware"]["nvidia_gpus"][0]["name"] == "NVIDIA H200"


def test_redact_text_removes_home_and_working_directory(monkeypatch: object) -> None:
    from causalneurotwin import provenance

    monkeypatch.setattr(provenance.Path, "home", classmethod(lambda cls: Path("/home/private")))  # type: ignore[attr-defined]
    monkeypatch.setattr(provenance.Path, "cwd", classmethod(lambda cls: Path("/work/private")))  # type: ignore[attr-defined]
    result = provenance.redact_text("/home/private/a and /work/private/b")
    assert result == "<home>/a and <working-directory>/b"
