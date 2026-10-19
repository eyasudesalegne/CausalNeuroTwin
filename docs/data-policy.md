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

## Scientific synchronization boundary
Never commit restricted review figures/archives, arrays, output bundles, private configuration, internal execution receipts or unrestricted environment dumps. Pseudonymous subject labels do not make participant-derived evidence anonymous.

The staged-blob scanner rejects private paths, common credential signatures, restricted result/review directories, binary scientific assets and large files. It reads the actual Git index even if an unsafe staged file has been replaced by a clean unstaged file. This conservative gate supplements explicit human disclosure review; it cannot infer consent or detect every sensitive metadata field.

Only synthetic configuration and incomplete nonidentifying experiment-record examples are distributed. Input and result manifest hashes and scheduler IDs require explicit disclosure review before export.
