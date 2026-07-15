"""Command-line interface for the CausalNeuroTwin repository foundation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causalneurotwin import __version__
from causalneurotwin.doctor import build_report, render_human, resolve_data_root, resolve_output_dir


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = build_report(
            data_root=resolve_data_root(args.data_root),
            output_dir=resolve_output_dir(args.output_dir),
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_human(report))
        return 0 if report.overall_status == "ready" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
