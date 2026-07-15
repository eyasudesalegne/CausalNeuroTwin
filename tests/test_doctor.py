from __future__ import annotations

import importlib.util
from pathlib import Path

from causalneurotwin.doctor import (
    _module_available,
    build_report,
    inspect_accelerator,
    inspect_data_root,
    inspect_output,
    render_human,
    resolve_data_root,
    resolve_output_dir,
)


def test_data_root_not_configured() -> None:
    status = inspect_data_root(None)
    assert status.configured is False
    assert status.status == "not_configured"


def test_data_root_ready_without_path_disclosure(tmp_path: Path) -> None:
    status = inspect_data_root(tmp_path)
    assert status.configured is True
    assert status.status == "ready"
    assert not hasattr(status, "path")


def test_data_root_missing_and_not_directory(tmp_path: Path) -> None:
    assert inspect_data_root(tmp_path / "missing").status == "missing"
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    assert inspect_data_root(file_path).status == "not_a_directory"


def test_output_is_writable(tmp_path: Path) -> None:
    status = inspect_output(tmp_path / "output")
    assert status.writable is True
    assert status.status == "ready"
    assert list((tmp_path / "output").iterdir()) == []


def test_output_failure_when_parent_is_file(tmp_path: Path) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("x", encoding="utf-8")
    status = inspect_output(parent / "child")
    assert status.writable is False
    assert status.status == "not_writable"


def test_accelerator_without_torch() -> None:
    status = inspect_accelerator(False)
    assert status.status == "torch_not_installed"
    assert status.cuda_device_count == 0


def test_resolve_paths_from_explicit_and_environment(tmp_path: Path, monkeypatch: object) -> None:
    explicit = tmp_path / "explicit"
    assert resolve_data_root(explicit) == explicit
    assert resolve_output_dir(explicit) == explicit

    monkeypatch.setenv("CAUSALNEUROTWIN_DATA_ROOT", str(tmp_path / "data"))  # type: ignore[attr-defined]
    monkeypatch.setenv("CAUSALNEUROTWIN_OUTPUT_ROOT", str(tmp_path / "runs"))  # type: ignore[attr-defined]
    assert resolve_data_root() == tmp_path / "data"
    assert resolve_output_dir() == tmp_path / "runs"

    monkeypatch.delenv("CAUSALNEUROTWIN_DATA_ROOT")  # type: ignore[attr-defined]
    monkeypatch.delenv("CAUSALNEUROTWIN_OUTPUT_ROOT")  # type: ignore[attr-defined]
    assert resolve_data_root() is None
    assert resolve_output_dir() == Path(".causalneurotwin-doctor")


def test_module_available_handles_invalid_spec(monkeypatch: object) -> None:
    def fail(_: str) -> None:
        raise ValueError("invalid")

    monkeypatch.setattr(importlib.util, "find_spec", fail)  # type: ignore[attr-defined]
    assert _module_available("broken") is False


def test_complete_report_and_human_render(tmp_path: Path) -> None:
    report = build_report(data_root=None, output_dir=tmp_path)
    payload = report.to_dict()
    rendered = render_human(report)
    assert report.package == "causalneurotwin"
    assert report.overall_status == "ready"
    assert payload["data_root"]["status"] == "not_configured"
    assert "numpy" in payload["optional_dependencies"]
    assert "torch" in payload["optional_dependencies"]
    assert "Optional dependencies:" in rendered
    assert "Overall: ready" in rendered
