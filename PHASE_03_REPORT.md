# Phase 03 report — Public dataset registration

## Dataset selected

- OpenNeuro accession: `ds004024`
- pinned version: `1.0.1`
- DOI: `10.18112/openneuro.ds004024.v1.0.1`
- licence: `CC0`
- expected participants: `13`

## Implemented

- strict versioned dataset-registry schema;
- registered identity, source, licence, access, BIDS, modality, ethics, intended-use, and prohibited-use metadata;
- environment-variable or explicit-root resolution without path disclosure;
- local identity and top-level metadata validation;
- participant-table and subject-directory alignment checks;
- T1w filename-layout check for every declared participant;
- non-sensitive modality inventory;
- metadata checksums and generated-output checksums;
- JSON, Markdown, and TSV reports;
- synthetic metadata-only fixture generator for tests and CI;
- CLI command `causalneurotwin dataset validate`;
- tests for valid registration, mismatched metadata, missing subjects, missing T1w layout, privacy, and CLI behavior.

## Explicitly not implemented

- automated download of the full dataset;
- full BIDS validation;
- image or EEG payload readability;
- DWI b-value or b-vector validation;
- session-level inclusion/exclusion;
- scientific quality control;
- participant selection;
- preprocessing or modelling.

## Phase gate

Phase 03 passes when the registry and synthetic fixture validate in CI, reports disclose no absolute data path or participant attributes, and the repository continues to reject tracked participant-level data.
