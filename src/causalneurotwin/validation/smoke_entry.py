"""Guarded synthetic smoke entry; independent of restricted subject configuration."""

import argparse
import json
from pathlib import Path

from causalneurotwin.checksums import sha256_file
from causalneurotwin.provenance import collect_git_provenance, collect_package_versions

from .artifacts import compute_guard, safe, write
from .readiness.dependencies import dependency_report


def execute(config_path, output_root, run_id, *, runner=None):
    compute_guard()
    config_path = safe(config_path)
    root = safe(output_root)
    c = json.loads(config_path.read_text())
    if (
        set(c) != {"schema_version", "example_kind", "wp2", "wp3"}
        or c["example_kind"] != "synthetic_algorithm_controls"
    ):
        raise ValueError("Only public synthetic algorithm controls accepted")
    if not run_id.startswith("synthetic_") or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("Explicit synthetic run ID")
    if (root / run_id).exists():
        raise FileExistsError(root / run_id)
    deps = dependency_report("smoke")
    if runner is None:
        from .readiness.smoke import run_smoke

        runner = run_smoke
    root.mkdir(parents=True, exist_ok=True)
    out = root / run_id
    out.mkdir()
    before = sha256_file(config_path)
    try:
        write(
            out / "provenance.json",
            {
                "mode": "synthetic_smoke",
                "config_sha256": before,
                "git": collect_git_provenance(Path(__file__).resolve().parents[3]),
                "software_versions": collect_package_versions(),
                "dependencies": deps,
                "dwi_header_guard": None,
                "participant_arrays_loaded": False,
            },
        )
        result = runner(c, out)
        if result.get("integration_steps") != 0:
            raise RuntimeError("Synthetic smoke must perform zero integration")
        if sha256_file(config_path) != before:
            raise RuntimeError("Config changed")
        write(
            out / "summary.json",
            {
                "execution": "completed",
                "assessment": result,
                "biological_validation": "not_established",
            },
        )
        write(
            out / "bundle_manifest.json",
            {
                "sha256": {
                    p.relative_to(out).as_posix(): sha256_file(p)
                    for p in out.rglob("*")
                    if p.is_file()
                }
            },
        )
    except Exception as error:
        write(
            out / "failure.json",
            {"execution": "failed", "error": type(error).__name__, "reason": str(error)},
        )
        raise
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Synthetic zero-integration TVB smoke; allocated node only"
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    execute(a.config, a.output_root, a.run_id)


if __name__ == "__main__":
    main()
