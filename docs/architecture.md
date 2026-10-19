# Architecture
The original CLI, doctor, typed run configuration, immutable lifecycle, checksum and privacy-aware provenance modules remain the engineering foundation.

Scientific algorithms are installed under a single namespace:
- validation.anatomy: MRI/GIFTI physical geometry, mesh metrics, evidence/review utilities.
- validation.anatomy_addendum: boundary-distance, cluster, parcel and mesh-localization methods reusing anatomy.
- validation.readiness: diffusion/connectome and TVB history/numerical methods.
- validation.structural: reviewed source selections, exact region order, mapping/network alternatives and held-out metrics.
- validation.interfaces: optional SimNIBS normal-projection/forward boundary.
- validation.artifacts: shared exclusive writes, path and allocated-node guards, reusing foundation checksums.
- experiment_records and publication: strict disclosure/linkage validation and inspection of exact staged Git blobs.

Imports use the installed package. No code imports the sibling sealed-release directories or mutates sys.path to load private scientific code. Subject-specific Slurm/configuration/deployment wrappers remain outside this migration. The source-port inventory documents exclusions.

Dynamic scientific kernels retain their source mathematical algorithms; namespace changes, formatting and shared IO are migration changes. Strict typing/coverage claims for the foundation do not automatically extend to optional external-tool adapters. CI reports these scopes separately.

Scientific-runtime checks require explicit allocated-node approval. They are not part of ordinary CI. No integration or remote submission is triggered by package import.
