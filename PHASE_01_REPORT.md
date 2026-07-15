# Phase 01 completion report

## Implemented and tested

- installable package with `src/` layout;
- CLI entry point and version command;
- privacy-preserving `doctor` diagnostic;
- optional dependency and CUDA inspection;
- writable-output validation;
- forbidden-data and secret-file scanner;
- tests for import, version, CLI, diagnostic, scanner, and wheel build;
- lint, format, typing, coverage, build, and CI configuration;
- repository policies and non-production HPC templates.

## Implemented but environment-dependent

- CUDA device detection, which reports actual availability only when PyTorch and a compatible runtime are installed;
- Git tracked-file scanning, which falls back to working-tree scanning before a Git repository is initialised.

## Planned

- typed scientific configuration;
- immutable run directories;
- provenance and structured logging;
- public-dataset registration;
- all scientific and HPC production functionality.

## Out of scope for Phase 01

No participant data, neuroimaging processing, connectome construction, stimulation modelling, brain simulation, model training, clinical evaluation, or scaling benchmark is included.

## Verification commands

```bash
python -m pip install -e '.[dev]'
python scripts/verify_repository.py
ruff check .
ruff format --check .
mypy src scripts
pytest --cov=causalneurotwin --cov-report=term-missing
python -m build
causalneurotwin doctor --json
```

## Phase gate

Advance to Phase 02 only when all commands pass in a clean clone and the README remains consistent with actual functionality.
