# Configuration

Committed configurations must be non-sensitive, versioned, and validated.

- `run_contract.example.yaml` exercises the Phase 02 run lifecycle without participant data.
- `data/openneuro_ds004024.yaml` pins the Phase 03 public dataset registration to OpenNeuro ds004024 version 1.0.1.
- `project.example.toml` is retained as the Phase 01 diagnostic example.

Absolute private paths, credentials, and participant-level data are prohibited. Dataset locations are supplied by `CAUSALNEUROTWIN_DATA_ROOT` or an explicit CLI argument that is never written to reports.
