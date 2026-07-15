# Security Policy

## Supported versions

Only the latest tagged pre-release or release is supported.

## Private reporting

Do not open a public issue for:

- exposed credentials or tokens;
- participant-data leakage;
- controlled-data access problems;
- private-key exposure;
- arbitrary-code execution;
- unsafe container or scheduler behaviour.

Report privately to `e.beyene@ogr.iuc.edu.tr` and include the affected commit, reproduction steps, impact, and proposed mitigation if known.

## Scope

This policy covers software and data-handling security. It does not certify the software for clinical, diagnostic, or therapeutic use.

## Incident priorities

1. Stop further disclosure or unsafe execution.
2. Preserve evidence without redistributing sensitive material.
3. Revoke affected credentials.
4. Assess repository history and released artefacts.
5. Patch, test, document, and notify affected collaborators.
