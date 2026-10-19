# Scientific synchronization audit
Baseline: official main commit 99ac3df40cf20ed37b1f20a9fe7c45a6fb8d72f7 (2026-07-15). The original Windows validation workspace was not a Git checkout. A separate clean clone was used; its feature branch has not been committed, pushed or merged.

## Migrated
One canonical implementation of reusable anatomy/addendum, DWI/connectome, TVB numerical-history and structural-sensitivity kernels. Existing synthetic tests were adapted to package imports. Shared artifact helpers reuse foundation checksums. Optional forward interfaces contain no coil, atlas or leadfield assets.

## Kept restricted or pending manual review
Original subject inventories/configurations, input/result integrity records, real job receipts, internal paths/accounts, review exports/images, data arrays, frozen results, sealed source-release manifests, deployment helpers and subject-bound orchestration. Older readiness releases remain local archives rather than duplicate public implementations.

The per-file private classification report is deliberately outside this checkout because filenames and provenance can themselves be sensitive. Its categories are publishable source candidate, requires sanitization, and restricted. Publication candidates still undergo exact staged-content scanning and human approval.

## Known defect and migration boundary
Deployed readiness v1.2 stopped before submission because synthetic provenance referenced an absent DWI key. The new public smoke entry explicitly records no DWI and has a complete stub-runner regression. This is a local integration change, not a correction to the immutable deployment. Runtime verification and any source redeployment require separate approval.

## Reproducibility
No restricted historic run is attributed to this baseline or proposed future commit. An experiment record may claim execution linkage only when commit/clean state, versioned configuration, software, code/input integrity, job ID and result-manifest checksum were captured and disclosure-reviewed. Otherwise retain an unlinked/retrospective status internally.

CI checks synthetic algorithms independently from the engineering foundation. No participant data or unavailable scientific runtime is required for ordinary CI.

[Source-code port lineage](source-ports.json) records original source hashes and adapted public source hashes. It contains no scientific input/result manifests. All migrated module numerical literals were checked unchanged; new interfaces and portability fixes are separately tested.
