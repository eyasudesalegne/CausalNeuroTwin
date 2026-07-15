# CausalNeuroTwin

[![CI](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/eyasudesalegne/CausalNeuroTwin/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#project-status)

Physics-informed NeuroAI research software for subject-specific modelling of whole-brain responses to non-invasive brain stimulation.

## Project status

**Pre-alpha Phase 03 public-dataset registration.** The repository implements a reproducible software and run foundation and pins one public OpenNeuro dataset. It does not yet validate scientific payloads, preprocess neuroimaging, simulate brain dynamics, or train a NeuroAI model.

### Implemented and tested

- installable Python package using a `src/` layout;
- `causalneurotwin doctor` environment diagnostic;
- strict typed experiment configuration;
- immutable run identity, provenance, checksums, logs, failure markers, and safe resume;
- strict public-dataset registry schema;
- OpenNeuro `ds004024`, version `1.0.1`, pinned by DOI;
- local dataset identity and top-level metadata validation;
- participant-table and subject-directory alignment checks;
- anatomical T1w filename-layout checks;
- non-sensitive modality inventory and validation reports;
- repository scanner preventing tracked participant data, models, and secrets;
- unit, integration, negative-path, privacy, build, and CI checks.

### Planned, not implemented

- automated acquisition of the full dataset;
- official BIDS Validator execution;
- image, EEG, fMRI, or DWI payload-readability checks;
- subject/session inclusion and exclusion;
- anatomical and diffusion preprocessing;
- structural-connectome construction;
- SimNIBS electric-field modelling;
- The Virtual Brain or reference mechanistic simulation;
- NeuroAI model training, uncertainty, or clinical validation;
- measured multi-node or multi-GPU scaling.

No planned capability should be interpreted as already implemented.

## Scientific objective

The long-term research objective is to study the mapping

```text
subject anatomy + structural connectome + baseline brain state
+ stimulation field + stimulation protocol
                         ↓
time-resolved whole-brain response + calibrated uncertainty
```

## Installation

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

## Phase 03 dataset registration

CausalNeuroTwin registers OpenNeuro `ds004024` version `1.0.1`. Raw data remain outside Git.

```bash
causalneurotwin dataset validate \
  --registry configs/data/openneuro_ds004024.yaml \
  --dataset-root /path/to/ds004024-v1.0.1 \
  --output-dir runs/dataset-registration-ds004024
```

Alternatively, place the dataset under

```text
${CAUSALNEUROTWIN_DATA_ROOT}/openneuro/ds004024-v1.0.1/
```

and omit `--dataset-root`.

The Phase 03 command checks identity and layout only. A passing result does not mean the scientific payloads are BIDS-valid or analysis-ready.

## Repository verification

```bash
python scripts/verify_repository.py
make verify
```

## Data policy

Raw or participant-level MRI, diffusion MRI, fMRI, EEG, MEG, clinical data, credentials, and controlled-data metadata must never be committed. Local paths and participant attributes are omitted from generated registration reports. See [docs/data-policy.md](docs/data-policy.md).

## Responsible-use boundary

CausalNeuroTwin is research software. It is not a medical device, diagnostic system, treatment-selection system, stimulation controller, or substitute for qualified clinical judgement. It must not be used to apply stimulation to a human participant.

## Roadmap

Phase 04 will run full BIDS and payload validation and produce explicit subject/session inclusion and exclusion reports before any preprocessing begins.

## Contributing and governance

Read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Cite the exact repository version, Git commit, and dataset version used.

## Licence

Original repository code is licensed under the Apache License 2.0. External software and datasets retain their own licences and data-use conditions.
