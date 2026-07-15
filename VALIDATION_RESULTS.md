# Phase 03 validation results

Validation was executed on **15 July 2026** in a clean Linux environment.

## Tested environment

- Operating system: Linux x86_64
- Python: 3.13.5
- Package version: 0.3.0a1
- Test runner: pytest 8.4.2

Python 3.11 and 3.12 are configured in GitHub Actions and must be confirmed by the Phase 03 pull-request CI before merge.

## Results

| Check | Result |
|---|---|
| Editable installation with runtime and development dependencies | Passed |
| Forbidden-data repository verification | Passed |
| Ruff lint | Passed |
| Ruff formatting check | Passed |
| Strict mypy check | Passed |
| Automated tests | **78 passed** |
| Branch-aware coverage | **92.86%** |
| Source distribution build | Passed |
| Wheel build | Passed |
| `causalneurotwin doctor --json` | Passed |
| Phase 02 successful run lifecycle | Passed |
| Phase 03 metadata-only fixture generation | Passed |
| Phase 03 dataset registration validation | Passed |
| Generated report SHA-256 verification | Passed |
| Validation-report private-path check | Passed |
| Participant-attribute omission check | Passed |

## Verified Phase 03 outputs

The metadata-only validation run produced:

```text
dataset_validation.json
dataset_validation.md
registry_snapshot.json
subject_modality_inventory.tsv
source_metadata_checksums.sha256
checksums.sha256
```

Every generated-output checksum verified successfully.

## Correct interpretation

These results validate the dataset-registry schema, ds004024 identity contract, local metadata and layout checks, privacy controls, report generation, and CLI behavior. They do not validate the real OpenNeuro payloads, official BIDS compliance, NIfTI or EEG readability, DWI gradients, session completeness, scientific quality, preprocessing, connectome construction, stimulation modelling, NeuroAI performance, clinical utility, or HPC scalability.
