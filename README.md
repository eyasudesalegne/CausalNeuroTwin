# CausalNeuroTwin

[![CI](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#project-status)

Physics-informed NeuroAI research software for subject-specific modelling of whole-brain responses to non-invasive brain stimulation.

## Project status

**Pre-alpha repository foundation.** This release establishes the software-engineering, governance, testing, and HPC-development foundation. It does not yet implement the scientific CausalNeuroTwin pipeline.

### Implemented and tested

- installable Python package using a `src/` layout;
- `causalneurotwin doctor` environment diagnostic;
- optional-dependency and accelerator detection;
- privacy-preserving data-root status reporting;
- writable-output verification;
- forbidden-data and secret-file repository scanner;
- unit tests, type checking, linting, coverage, wheel build, and CI;
- contribution, governance, security, citation, and data-handling policies;
- non-production Slurm templates for future CPU and GPU validation.

### Planned, not implemented

- ingestion of HCP or OpenNeuro datasets;
- BIDS validation and subject-level quality control;
- anatomical and diffusion-MRI preprocessing;
- structural-connectome construction;
- SimNIBS electric-field modelling;
- The Virtual Brain integration;
- mechanistic response simulation;
- graph neural networks or neural operators;
- model training, uncertainty estimation, or clinical validation;
- measured multi-node or multi-GPU scaling.

No documentation in this repository should be interpreted as evidence that a planned capability already exists.

## Scientific objective

The long-term research objective is to study the mapping

```text
subject anatomy + structural connectome + baseline brain state
+ stimulation field + stimulation protocol
                         ↓
time-resolved whole-brain response + calibrated uncertainty
```

The project is intended to combine public or properly authorised neuroimaging data, electromagnetic-field modelling, connectome-constrained dynamical simulation, and NeuroAI methods. Phase 01 contains only the repository foundation required to develop those components reproducibly.

## Installation

Python 3.11–3.13 is supported by the repository foundation. Scientific dependencies introduced in later phases may narrow this range.

```bash
git clone https://github.com/eyasudesalegne/CausalNeuroTwin.git
cd CausalNeuroTwin

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Environment diagnostic

```bash
causalneurotwin doctor
```

Machine-readable output:

```bash
causalneurotwin doctor --json
```

Optional local configuration:

```bash
export CAUSALNEUROTWIN_DATA_ROOT=/path/outside/the/repository
causalneurotwin doctor --output-dir ./runs/doctor-check
```

The diagnostic reports whether a data root is configured and usable, but it does not print the private path.

## Repository verification

```bash
python scripts/verify_repository.py
```

The verification script rejects common participant-data, model-checkpoint, credential, and private-key file patterns. It is a safety control, not a substitute for human review.

## Development checks

```bash
ruff check .
ruff format --check .
mypy src scripts
pytest --cov=causalneurotwin --cov-report=term-missing
python -m build
```

Or run the consolidated target:

```bash
make verify
```

## Repository layout

```text
src/causalneurotwin/   Installable package
configs/               Version-controlled non-sensitive configuration examples
docs/                  Scope, architecture, data policy, and HPC-development notes
scripts/               Repository verification utilities
tests/                 Unit, CLI, scanner, and build tests
hpc/                    Non-production Slurm templates
.github/                CI, review, ownership, and issue templates
```

## Data policy

Raw or participant-level MRI, diffusion MRI, fMRI, EEG, MEG, clinical data, credentials, and controlled-data metadata must never be committed. Local data must remain outside the repository and be referenced through environment variables or local untracked configuration. See [docs/data-policy.md](docs/data-policy.md).

## Responsible-use boundary

CausalNeuroTwin is research software. It is not a medical device, diagnostic system, treatment-selection system, stimulation controller, or substitute for qualified clinical judgement. It must not be used to apply stimulation to a human participant.

## Roadmap

The next phase will add a typed configuration model, immutable run directories, provenance capture, structured logging, checksums, and failure-state handling. Scientific data processing starts only after those contracts are tested.

## Contributing and governance

Read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Until an archived scientific release exists, cite the exact repository version and Git commit used.

## Licence

Original repository code is licensed under the Apache License 2.0. External software and datasets retain their own licences and data-use conditions.
