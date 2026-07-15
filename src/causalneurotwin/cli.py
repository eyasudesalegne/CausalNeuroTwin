"""Command-line interface for CausalNeuroTwin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causalneurotwin import __version__
from causalneurotwin.config import ConfigError, load_config
from causalneurotwin.datasets import (
    DatasetRegistryError,
    load_dataset_registry,
    resolve_dataset_root,
    validate_local_dataset,
    write_dataset_validation_report,
)
from causalneurotwin.doctor import build_report, render_human, resolve_data_root, resolve_output_dir
from causalneurotwin.run_contract import RunContractError, execute_contract_demo


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="causalneurotwin",
        description="CausalNeuroTwin pre-alpha research software.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect the local runtime without exposing private data paths.",
    )
    doctor.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional local data root. The path is never printed.",
    )
    doctor.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for a temporary write test.",
    )
    doctor.add_argument("--json", action="store_true", help="Emit JSON output.")

    contract = subparsers.add_parser(
        "run-contract",
        help="Validate configuration and create a non-scientific Phase 02 run bundle.",
    )
    contract.add_argument(
        "--config", type=Path, required=True, help="Validated YAML configuration."
    )
    contract.add_argument(
        "--output-root", type=Path, default=Path("runs"), help="Root for immutable run folders."
    )
    contract.add_argument(
        "--run-id", default=None, help="Explicit deterministic run ID; otherwise one is generated."
    )
    contract.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete run with matching configuration.",
    )
    contract.add_argument(
        "--exercise-failure",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    dataset = subparsers.add_parser(
        "dataset",
        help="Register and validate public datasets without committing participant data.",
    )
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_subparsers.add_parser(
        "validate",
        help="Validate a pinned dataset registry against a local copy.",
    )
    validate.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Versioned dataset registry YAML.",
    )
    validate.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Explicit local dataset root. The path is never written to reports.",
    )
    validate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for non-sensitive validation reports.",
    )
    validate.add_argument("--json", action="store_true", help="Emit validation JSON to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        report = build_report(
            data_root=resolve_data_root(arguments.data_root),
            output_dir=resolve_output_dir(arguments.output_dir),
        )
        if arguments.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_human(report))
        return 0 if report.overall_status == "ready" else 1
    if arguments.command == "run-contract":
        try:
            config = load_config(arguments.config)
            run_dir = execute_contract_demo(
                config=config,
                config_path=arguments.config,
                output_root=arguments.output_root,
                run_id=arguments.run_id,
                resume=arguments.resume,
                command_line=["causalneurotwin", *(argv or sys.argv[1:])],
                fail_after_start=arguments.exercise_failure,
            )
        except (ConfigError, RunContractError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Run bundle completed: {run_dir.name}")
        return 0
    if arguments.command == "dataset" and arguments.dataset_command == "validate":
        try:
            registry = load_dataset_registry(arguments.registry)
            dataset_root, root_source = resolve_dataset_root(
                registry,
                explicit_root=arguments.dataset_root,
            )
            validation = validate_local_dataset(
                registry,
                dataset_root,
                root_source=root_source,
            )
            write_dataset_validation_report(
                registry=registry,
                validation=validation,
                dataset_root=dataset_root,
                registry_path=arguments.registry,
                output_dir=arguments.output_dir,
            )
        except (DatasetRegistryError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if arguments.json:
            print(json.dumps(validation.to_dict(), indent=2, sort_keys=True))
        else:
            print(
                f"Dataset registration {validation.status}: "
                f"{validation.dataset_id} v{validation.dataset_version}"
            )
            print(f"Reports: {arguments.output_dir.name}")
        return 0 if validation.status == "pass" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
