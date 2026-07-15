from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.build
def test_wheel_builds(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("causalneurotwin-*.whl"))
    assert len(wheels) == 1
