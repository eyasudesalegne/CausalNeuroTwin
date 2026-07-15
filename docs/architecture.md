# Architecture

## Phase 01 boundaries

The package currently contains only three stable concerns:

1. **CLI** — user-facing command dispatch.
2. **Doctor** — privacy-preserving runtime inspection.
3. **Repository safety** — detection of prohibited artefacts.

Scientific modules are intentionally absent until their data contracts, validation criteria, and provenance requirements are specified.

## Design principles

- install the package rather than manipulating `PYTHONPATH`;
- keep external-tool integration behind explicit adapters;
- use typed inputs and outputs;
- keep configuration outside algorithm code;
- fail visibly on invalid or incomplete output;
- distinguish implemented behavior from planned architecture;
- keep raw participant data outside Git;
- measure performance rather than assume scaling.

## Planned progression

```text
Phase 01: package and safety foundation
Phase 02: configuration, run contract, provenance, and logging
Phase 03: public-dataset registry and validation
Phase 04+: connectome, stimulation, simulation, learning, and HPC scaling
```

## Dataset registry layer

Phase 03 adds a strict registry that separates stable public metadata from local data placement. Registry entries pin dataset identity, version, DOI, licence, access class, BIDS version, expected top-level metadata, modalities, intended use, and prohibited use. Local validation emits only a path fingerprint and non-sensitive inventory; it never writes an absolute dataset root.

This layer does not replace the official BIDS Validator or payload-level quality control.
