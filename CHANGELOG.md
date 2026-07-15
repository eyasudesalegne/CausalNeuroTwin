# Changelog

All notable changes are recorded here. The project follows semantic versioning once a stable public API exists.

## [Unreleased]

### Planned

- Public dataset registry and non-sensitive manifest validation.

## [0.2.0-alpha.1] - 2026-07-15

### Added

- Strict typed YAML configuration with schema pinning and unknown-key rejection.
- Project, reproducibility, data, simulation, model, training, evaluation, HPC, and output contracts.
- Immutable run identities and atomic artefact writes.
- Privacy-aware Git, command, environment, scheduler, package, CPU, memory, and GPU provenance.
- Independent random-seed recording.
- Input manifests and streaming SHA-256 output checksums.
- Human-readable and JSONL event logs.
- `RUNNING`, `RUN_COMPLETE`, and `RUN_FAILED` lifecycle state.
- Safe resume for incomplete runs with matching configuration.
- Non-scientific `causalneurotwin run-contract` validation command.
- Phase 02 lifecycle, negative-path, privacy, checksum, and configuration tests.

### Not included

- Participant-data processing, mechanistic simulation, model training, or performance claims.

## [0.1.0-alpha.1] - 2026-07-15

### Added

- Installable pre-alpha Python package.
- `causalneurotwin doctor` diagnostic command.
- Privacy-preserving data-root and writable-output checks.
- Optional scientific-dependency and accelerator detection.
- Forbidden-data and secret-file scanner.
- Unit, CLI, scanner, and wheel-build tests.
- Ruff, mypy, pytest, coverage, pre-commit, and GitHub Actions configuration.
- Repository documentation, governance, security, citation, and HPC templates.
