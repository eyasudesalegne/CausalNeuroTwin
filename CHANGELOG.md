# Changelog

All notable changes are recorded here. The project follows semantic versioning once a stable public API exists.

## [Unreleased]

### Planned

- Public dataset registry and non-sensitive manifest validation.

## [0.3.0-alpha.1] - 2026-07-15

### Added

- Strict public-dataset registry schema.
- OpenNeuro ds004024 version 1.0.1 registration pinned by DOI.
- Environment-variable and explicit-root dataset resolution without path disclosure.
- Local dataset identity, top-level metadata, participant-table, subject-directory, and T1w layout validation.
- Non-sensitive subject modality inventory.
- JSON, Markdown, TSV, source-metadata checksum, and output checksum reports.
- Metadata-only synthetic ds004024 fixture generator for tests and CI.
- `causalneurotwin dataset validate` CLI command.
- Phase 03 tests for metadata mismatch, missing subjects, missing T1w layout, privacy, registry validation, and CLI behavior.

### Not included

- Full BIDS validation, payload readability, subject selection, preprocessing, simulation, or model training.

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
