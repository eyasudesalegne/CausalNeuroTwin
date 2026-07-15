# CausalNeuroTwin

[![CI](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#project-status)

Physics-informed NeuroAI research software for subject-specific modelling of whole-brain responses to non-invasive brain stimulation.

## Project status

**Pre-alpha Phase 02 engineering foundation.** The repository now implements validated configuration, privacy-aware provenance, immutable run directories, checksums, lifecycle markers, and safe resume for incomplete runs. It does not yet implement the scientific CausalNeuroTwin pipeline.

### Implemented and tested

- installable Python package using a `src/` layout;
- `causalneurotwin doctor` environment diagnostic;
- strict typed YAML configuration with rejection of missing and unknown fields;
- independent seed policy and explicit SI time-unit fields;
- immutable run identity and atomic run artefacts;
- `RUNNING`, `RUN_COMPLETE`, and `RUN_FAILED` lifecycle markers;
- safe resume only for incomplete runs with matching configuration;
- privacy-aware environment, scheduler, package, hardware, Git, and command provenance;
- streaming SHA-256 input and output checksums;
- durable human-readable logs and JSONL events;
- forbidden-data and secret-file repository scanner;
- unit, integration, negative-path, type, lint, coverage, wheel-build, and CI checks;
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

Phase 02 contains only the reproducibility and lifecycle contracts required to develop those components safely.

## Installation

Python 3.11–3.13 is supported by the engineering foundation. Scientific dependencies introduced later may narrow this range.

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
causalneurotwin doctor --json
```

## Phase 02 run-contract validation

```bash
causalneurotwin run-contract \
  --config configs/run_contract.example.yaml \
  --output-root runs \
  --run-id phase02-validation
```

This command validates configuration, provenance, logging, checksums, lifecycle state, and output immutability. It intentionally performs no scientific simulation or model training. See [docs/run-contract.md](docs/run-contract.md).

## Repository verification

```bash
python scripts/verify_repository.py
make verify
```

## Repository layout

```text
src/causalneurotwin/   Installable package and run-contract implementation
configs/               Version-controlled non-sensitive configuration examples
docs/                  Scope, architecture, data policy, and run-contract documentation
scripts/               Repository verification utilities
tests/                 Unit, integration, lifecycle, privacy, and build tests
hpc/                    Non-production Slurm templates
.github/                CI, review, ownership, and issue templates
```

## Data policy

Raw or participant-level MRI, diffusion MRI, fMRI, EEG, MEG, clinical data, credentials, and controlled-data metadata must never be committed. Local data must remain outside the repository and be referenced through environment variables or local untracked configuration. See [docs/data-policy.md](docs/data-policy.md).

## Responsible-use boundary

CausalNeuroTwin is research software. It is not a medical device, diagnostic system, treatment-selection system, stimulation controller, or substitute for qualified clinical judgement. It must not be used to apply stimulation to a human participant.

## Roadmap

The next phase will register and validate one public dataset without committing participant-level data. Scientific processing begins only after dataset identity, licence, access, manifest, and subject-level QC are explicit.

## Contributing and governance

Read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Until an archived scientific release exists, cite the exact repository version and Git commit used.

## Licence

Original repository code is licensed under the Apache License 2.0. External software and datasets retain their own licences and data-use conditions.
