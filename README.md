# CausalNeuroTwin
Research software for subject-specific brain reconstruction, computational validation and future stimulation-response prediction.

This development branch extends the Phase 02 engineering foundation with reusable anatomical, parcellation, diffusion/connectome and TVB numerical-verification modules. Restricted project reconstructions and computational QC runs exist separately. They are not distributed as repository data.

## Implemented engineering
The original configuration, immutable run-contract, provenance, checksums and diagnostic CLI remain available. Scientific code now lives once under `src/causalneurotwin/validation/`:
- physical MRI/GIFTI geometry, tissue-distance and mesh-quality calculations;
- independent-review handling and Brainnetome seed/completion distinctions;
- gradient-frame, motion-feasibility and endpoint/pass comparison helpers;
- precision-aware TVB history, delayed-coupling, monitor and numerical-observation checks;
- cortical-source, atlas-mapping and connectome-definition sensitivity methods;
- guarded synthetic smoke and optional SimNIBS forward interfaces;
- held-out metric/split safeguards, disclosure-reviewed experiment-record contracts and staged-blob publication checks.

## Scientific status
| Component | Status and boundary |
|---|---|
| Restricted reconstruction and anatomical QC | Computational runs completed in the research project; independent anatomical acceptance remains pending |
| DWI/connectome QC | Existing derivative/provenance audits completed; high motion and reconstruction uncertainty remain limitations |
| Gradient conventions | Prior apparent frame mismatch resolved through documented conventions; this does not establish edge truth |
| TVB history adapter | Implemented and tested on synthetic references; installed-runtime compatibility remains pending |
| Deployed TVB v1.2 smoke | Submission stopped during preflight: provenance accesses a DWI field absent from synthetic configuration; no job ran |
| This branch's smoke entry | Uses explicit synthetic provenance and has a stub-runner regression; it has not been deployed or runtime-validated |
| Intrinsic convergence | Eight-condition design prepared; real-network timestep/stability/initial-state testing remains pending |
| Structural sensitivity | Methods and synthetic tests implemented; subject-level execution and held-out prediction remain pending |
| Experimental stimulation | Original Cool-B70 demonstration is frozen; matched C-B60 pose/dose evidence is incomplete |
| Measured EEG | Available in approved research storage; artifact handling, registration and predictive validation are separate gates |
| Physiological prediction | No claim of subject-specific biological calibration or validated measured-response prediction |

Successful execution, finite signals, derivative agreement and improved fit do not by themselves establish anatomical or physiological validity.

## Installation and safe tests
Python 3.11–3.13 supports the engineering foundation. Optional scientific-runtime support may be narrower. No participant dataset or scientific result is included.
```bash
python -m pip install -e ".[dev,scientific]"
python -m pytest --ignore=tests/scientific
python -m pytest tests/scientific -m "not allocated"
python scripts/verify_repository.py
```
TVB/SimNIBS runtime checks are separate and require approved installed environments. Ordinary tests use generated nonparticipant fixtures. CI does not submit jobs or integrate TVB.

The Phase 02 CLI remains:
```bash
causalneurotwin doctor --json
causalneurotwin run-contract --config configs/run_contract.example.yaml --output-root runs --run-id example
```
The run-contract command exercises engineering lifecycle contracts only.

## Reproducibility and publication
See [scientific methods](docs/scientific-validation.md), [scope](docs/current-scope.md), [migration audit](docs/scientific-sync.md), [experiment records](docs/experiment-records.md), and [predictive protocol](docs/predictive-validation.md).

Keep private configuration, real input manifests, scheduler metadata and result bundles outside Git. Publish sanitized experiment records only after explicit disclosure review. Old uncommitted runs must never be assigned a later Git SHA retrospectively.

## Layout
```text
src/causalneurotwin/validation/  Canonical reusable scientific code
configs/                       Synthetic controls and empty record examples
tests/scientific/              Nonparticipant tests and separately marked runtime checks
docs/                          Methods, status, limitations and data policy
hpc/                           Non-production templates; no submission helper
scripts/                       Repository and staged-publication verification
```

CausalNeuroTwin is research software, not a medical device, stimulation controller or clinical decision system.
