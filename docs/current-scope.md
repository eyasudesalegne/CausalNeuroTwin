# Current scope
The Phase 02 engineering contracts remain implemented. This branch additionally integrates reusable scientific verification and sensitivity modules.

## Completed implementation
Anatomical physical-coordinate, mesh, review and surface-distance methods; atlas mapping diagnostics; gradient/motion/connectome methods; TVB history/numerical observers; structural source/atlas/network sensitivity; observation interfaces; predictive metrics and leakage checks; publication and provenance contracts.

## Completed computational experiments outside this checkout
The project has subject-specific reconstructions, anatomical QC bundles, diffusion/connectome provenance audits and frozen demonstration/control simulations. They remain in approved restricted storage and are not bundled here. Computational completion does not imply independent anatomical acceptance, reliable individual edges or physiological calibration.

## Pending
Installed-runtime TVB history compatibility; real-network intrinsic convergence and stability; independent anatomical review; measured-electrode/MRI registration acceptance; matched experimental C-B60 pose and dose; artifact-controlled, held-out EEG prediction.

The deployed v1.2 smoke did not run: preflight identified a missing DWI key in provenance generation. A separate source integration entry uses explicit synthetic provenance and a regression test, but is not a deployed correction or evidence of runtime compatibility.

No production reconstruction pipeline, raw participant dataset, newly generated TMS field, trained predictive model or validated clinical capability is distributed here.
