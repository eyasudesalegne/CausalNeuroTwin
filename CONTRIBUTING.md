# Contributing

## Workflow

1. Open or select an issue before substantial work.
2. Create a focused branch from `main`.
3. Keep commits small and descriptive.
4. Add tests and documentation with the implementation.
5. Run `make verify` before opening a pull request.
6. Use the pull-request template and disclose scientific, data, and performance implications.
7. Do not merge until required checks and review are complete.

## Branch names

- `feat/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`
- `perf/<short-name>`
- `refactor/<short-name>`

## Scientific changes

A change to a scientific method must document:

- scientific rationale;
- equations or algorithm reference;
- assumptions and units;
- input and output contracts;
- expected numerical behaviour;
- validation and numerical tolerances;
- reproducibility impact;
- performance impact;
- limitations.

## Data rules

Never commit participant-level data, credentials, access tokens, controlled metadata, private paths, raw repository downloads, or unrestricted model checkpoints. Tests must use synthetic or explicitly redistributable fixtures.

## Definition of done

A contribution is complete only when code, tests, documentation, changelog, provenance implications, and failure behaviour are consistent.
