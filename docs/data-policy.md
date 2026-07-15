# Data policy

## Core rule

Git stores source code, schemas, documentation, synthetic fixtures, and non-sensitive manifests. It does not store participant-level research data.

## Prohibited repository content

- MRI, diffusion MRI, fMRI, EEG, MEG, or clinical files;
- direct identifiers or uncontrolled subject metadata;
- credentials, access tokens, private keys, or private configuration;
- private absolute paths;
- raw HCP or OpenNeuro downloads;
- scientific array stores or trained checkpoints without explicit release review.

## Local data

Local data roots must be configured through environment variables or untracked local configuration. The `doctor` command reports status without printing the path.

## Future dataset registration

Before using a dataset, the project will record its exact version, source, licence, data-use terms, access date, checksums, modalities, exclusions, permitted derivatives, and retention rules.

## Release review

Derived data or model weights will require a separate review of source licences, re-identification risk, sensitive metadata, subgroup harm, and dual-use implications.
