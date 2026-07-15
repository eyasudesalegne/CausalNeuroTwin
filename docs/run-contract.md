# Phase 02 run contract

Phase 02 introduces a typed configuration and an immutable, restart-aware run directory. The command validates software engineering behavior only. It does not process participant data, simulate a brain, or train a model.

## Execute

```bash
causalneurotwin run-contract \
  --config configs/run_contract.example.yaml \
  --output-root runs \
  --run-id phase02-validation
```

A successful run contains:

```text
run_identity.json
resolved_config.yaml
environment.json
input_manifest.json
provenance.json
events.jsonl
run.log
stdout.log
metrics.json
checksums.sha256
RUN_COMPLETE
```

`RUNNING` exists only while work is active. A failure produces `failure.json`, `stderr.log`, and `RUN_FAILED`; it never produces `RUN_COMPLETE`.

## Configuration guarantees

The loader rejects:

- missing required sections or fields;
- unknown fields;
- unsupported schema versions;
- invalid seeds and resource counts;
- time values without explicit seconds-based field names;
- non-positive simulation time steps;
- a time step larger than duration;
- enabled training without positive epochs, batch size, and learning rate;
- absolute or parent-traversing manifest paths;
- unsupported checksum algorithms.

The simulation, model, training, and evaluation sections are reserved contracts. They are disabled in the Phase 02 example because those scientific capabilities do not yet exist.

## Provenance and privacy

The run records:

- Git commit and dirty state when Git is available;
- package, Python, platform, CPU, memory, GPU, and selected dependency information;
- a hashed host fingerprint rather than the hostname;
- an allowlisted subset of scheduler variables;
- redacted command-line arguments;
- independent random seeds;
- input and output checksums.

It does not record usernames, home-directory paths, access tokens, credentials, or participant identifiers.

## Immutability and resume

A completed run cannot be resumed or overwritten. An incomplete run can be resumed only when its configuration digest exactly matches the original run identity. Resume appends a new attempt record while preserving prior logs and failure evidence.

Phase 02 does not promise general checkpoint recovery for scientific arrays because those arrays are not implemented yet. It establishes the lifecycle contract future commands must use.
