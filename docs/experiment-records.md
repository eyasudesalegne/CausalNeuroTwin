# Reproducibility and disclosure
Use configs/experiment_record.example.json as an EMPTY draft template. It intentionally contains no actual scheduler identifier, private path, scientific input hash, result hash or historical acceptance.

A completed public record requires:
1. The exact clean Git commit present at execution.
2. A versioned configuration and its SHA256.
3. Explicit software versions.
4. Code and input-integrity manifest SHA256.
5. Actual Slurm job ID and result-manifest SHA256.
6. An explicit completed disclosure review covering the exported fields.

Keep full execution provenance and restricted manifests on approved research infrastructure. A hash may still link to restricted records and is not automatically approved for publication. Do not export environment dumps, hostnames, usernames, absolute input paths or participant acquisition identifiers.

validate_experiment_record rejects unreviewed internal metadata and incomplete completed records. A later commit cannot be invented as the source of an old uncommitted run. Historical source hashes may support internal retrospective linkage; they are not execution-time Git provenance.

The proposed commit has no SHA until the user authorizes and creates it. Freeze configuration and capture the actual SHA at later approved execution; do not insert placeholders that look like genuine evidence.
