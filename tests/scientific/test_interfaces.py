import json
from pathlib import Path

import numpy as np
import pytest

from causalneurotwin.validation import smoke_entry
from causalneurotwin.validation.artifacts import sha
from causalneurotwin.validation.interfaces.simnibs import project_normal_gain


def test_normal_projection_is_not_fem():
    L = np.arange(12.0).reshape(2, 2, 3)
    n = np.array([[1, 0, 0], [0, 0, 2]])
    x = project_normal_gain(
        L, n, frame="native_scanner_RAS_mm", channel_names=["A", "B"], expected_channels=["A", "B"]
    )
    assert np.array_equal(x, L[:, [0, 1], [0, 2]])


@pytest.mark.parametrize("change", ["frame", "order", "normal"])
def test_forward_geometry_guards(change):
    L = np.zeros((2, 2, 3))
    n = np.ones((2, 3))
    frame = "native_scanner_RAS_mm"
    channels = ["A", "B"]
    if change == "frame":
        frame = "unverified"
    elif change == "order":
        channels.reverse()
    else:
        n[0] = 0
    with pytest.raises(ValueError):
        project_normal_gain(L, n, frame=frame, channel_names=channels, expected_channels=["A", "B"])


def test_smoke_provenance_without_dwi_reaches_stub_runner_and_seals(tmp_path, monkeypatch):
    """Regression for the deployed release's preflight-blocking KeyError; no TVB imports."""
    monkeypatch.setattr(smoke_entry, "compute_guard", lambda: None)
    monkeypatch.setattr(smoke_entry, "dependency_report", lambda mode: {"synthetic": True})
    config = (
        Path(__file__).resolve().parents[2] / "configs/scientific_validation.synthetic.example.json"
    )
    cp = tmp_path / "controls.json"
    cp.write_bytes(config.read_bytes())
    before = sha(cp)
    reached = []

    def runner(c, out):
        reached.append(True)
        assert "dwi" not in c
        return {"integration_steps": 0, "runtime_configuration": "synthetic_stub_only"}

    out = smoke_entry.execute(cp, tmp_path / "outputs", "synthetic_regression", runner=runner)
    p = json.loads((out / "provenance.json").read_text())
    assert p["dwi_header_guard"] is None and not p["participant_arrays_loaded"]
    assert reached and sha(cp) == before
    seal = json.loads((out / "bundle_manifest.json").read_text())["sha256"]
    assert all(sha(out / n) == h for n, h in seal.items())
    with pytest.raises(FileExistsError):
        smoke_entry.execute(cp, tmp_path / "outputs", "synthetic_regression", runner=runner)


def test_public_smoke_entry_refuses_login_before_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError):
        smoke_entry.execute(tmp_path / "absent.json", tmp_path / "outputs", "synthetic_blocked")
    assert not (tmp_path / "outputs").exists()
