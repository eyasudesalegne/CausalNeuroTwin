# Scientific validation methods and evidence standards
Software tests demonstrate implementation behavior on synthetic fixtures. They cannot establish anatomical correctness, tractography truth or physiological validity.

## Anatomy and atlas
Reuse existing reports before proposing new measurements. Physical coordinates require declared NIfTI qform/sform selection, units, scanner-RAS GIFTI transforms and explicit point-map direction. Boundary disagreement from categorical sampling is distinct from physical distance. Surface-normal offsets probe persistence; out-of-volume samples remain indeterminate.

Mesh shape metrics include mean ratio, scaled Jacobian and dihedrals. Thresholds are provisional triage rules; geometric quality is not an electric-field error estimate.

Direct correct-hemisphere atlas seeds, rejected samples and geodesic additions remain separate. Template closure candidates are hypotheses; only independently supported dispositions alter conservative masks. Unresolved-source omission and flagged-parcel ablation are sensitivity bounds, not accepted corrections. Preserve the original complete neural-region ordering, including nodes with zero direct cortical EEG gain.

## DWI and network definitions
Gradient comparison requires compatible documented coordinate frames and verified volume correspondence before angular statistics. Antipodal equivalence is appropriate for diffusion axes, but does not justify arbitrary rotations selected to minimize error. Motion/distortion limitations survive successful preprocessing.

Endpoint and pass definitions are alternatives, not truth labels. Common-edge controls retain explicitly chosen reference weights and paired lengths. A fixed reference scale avoids erasing strength differences. Existing strength/degree nulls may alter weight-length associations and cannot be described as pure topology controls. Do not infer false-positive/negative edge rates from endpoint/pass disagreement.

## TVB and observations
History storage is compared against exact expected float32 representations, without extra ULP slack. State/time/coupling oracles, active delayed inputs, circular wrapping and monitor boundaries are separate checks. Numerical tolerances and convergence criteria are provisional and are not physiological calibration.

The eight-condition numerical design varies timestep, initial condition/amplitude and warm-up without repeating earlier coupling/speed/null studies. Overflow, nonfinite states and unavailable checks must remain explicit.

Forward interfaces require verified native coordinates, exact channels and registration tied to the forward asset. Original regional normalization and physical source calibration are distinct issues. A finite forward matrix does not establish measured-electrode registration or calibrated voltages.

## Excluded deployment orchestration
This branch does not silently replace sealed HPC releases. Absolute input inventories, source/result manifests containing restricted execution metadata, private review forms and deployment wrappers need separate disclosure and portability review. The original smoke preflight defect remains recorded; source-only migration tests are not evidence that an installed TVB adapter works.
