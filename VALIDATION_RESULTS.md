# Phase 02 validation results

Validation was executed on **15 July 2026** in a clean virtual environment.

## Tested environment

- Operating system: Linux x86_64
- Python: 3.13.5
- Package version: 0.2.0a1
- Test runner: pytest 8.4.2

Python 3.11 and 3.12 are configured in GitHub Actions. They were not available in the local validation environment and must be confirmed by repository CI.

## Results

| Check | Result |
|---|---|
| Editable installation with runtime and development dependencies | Passed |
| Forbidden-data repository verification | Passed |
| Ruff lint | Passed |
| Ruff formatting check | Passed |
| Strict mypy check | Passed |
| Automated tests | **63 passed** |
| Branch-aware coverage | **94.00%** |
| Source distribution build | Passed |
| Wheel build | Passed |
| `causalneurotwin doctor --json` | Passed |
| Phase 02 successful run lifecycle | Passed |
| Intentional failure and safe resume lifecycle | Passed in automated tests |
| Run-bundle SHA-256 verification with `sha256sum -c` | Passed |
| Provenance private-path check | Passed |

## Verified Phase 02 output

The validation run produced `resolved_config.yaml`, `environment.json`, `input_manifest.json`, `provenance.json`, human and JSONL logs, `metrics.json`, `checksums.sha256`, and `RUN_COMPLETE`. Every entry in the checksum manifest verified successfully.

## Correct interpretation

These results validate configuration, provenance, logging, checksums, failure states, immutability, and resume behavior. They do not validate neuroimaging processing, connectome construction, brain stimulation modelling, mechanistic brain simulation, NeuroAI performance, clinical utility, or HPC scalability because those capabilities remain unimplemented.
