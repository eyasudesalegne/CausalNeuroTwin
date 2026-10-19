"""Gradient/motion provenance, matched sensitivity feasibility and uncertainty methods."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import matrix


def fsl_to_world(vectors, affine):
    a = np.asarray(affine, float)[:3, :3]
    length = np.linalg.norm(a, axis=0)
    if (length <= 0).any() or not np.isfinite(a).all():
        raise ValueError("Invalid affine")
    rotation = a / length
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Shear refused")
    v = np.asarray(vectors, float).copy()
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("Vectors must be N by 3")
    if np.linalg.det(a) > 0:
        v[:, 0] *= -1
    return v @ rotation.T


def gradients(bvals, bvecs, mrtrix, affine, nvol, confound_bvals, sources, original_files, options):
    b = np.asarray(bvals, float).reshape(-1)
    v = np.asarray(bvecs, float)
    m = np.asarray(mrtrix, float)
    if (
        v.shape != (3, nvol)
        or m.shape != (nvol, 4)
        or len(b) != nvol
        or len(confound_bvals) != nvol
        or len(original_files) != nvol
    ):
        raise ValueError("Gradient/confound/image volume count mismatch")
    if (
        not np.isfinite(b).all()
        or (b < 0).any()
        or not np.isfinite(v).all()
        or not np.isfinite(m).all()
    ):
        raise ValueError("Invalid gradient values")
    if not np.isfinite(confound_bvals).all() or not np.allclose(
        b, confound_bvals, atol=options["bval_tolerance"], rtol=0
    ):
        raise ValueError("Confound b-values not aligned")
    from pathlib import PurePosixPath

    names = lambda values: [PurePosixPath(str(v).replace("\\", "/")).name for v in values]
    source_names = names(sources)
    row_names = names(original_files)
    if (
        len(set(source_names)) != len(source_names)
        or set(row_names) != set(source_names)
        or not all(row_names)
    ):
        raise ValueError("Unresolved source-series provenance")
    dw = b > options["b0_max"]
    norms = np.linalg.norm(v.T, axis=1)
    mn = np.linalg.norm(m[:, :3], axis=1)
    if (
        not dw.any()
        or np.any(abs(norms[dw] - 1) > options["bvec_norm_tolerance"])
        or np.any(abs(mn[dw] - 1) > options["bvec_norm_tolerance"])
    ):
        raise ValueError("Nonunit diffusion directions")
    if not np.allclose(b, m[:, 3], atol=options["bval_tolerance"], rtol=0):
        raise ValueError("MRtrix b-values mismatch")
    world = fsl_to_world(v.T, affine)
    dots = np.sum(world[dw] * m[dw, :3], axis=1) / (norms[dw] * mn[dw])
    angle = np.degrees(
        np.arccos(np.clip(abs(dots), 0, 1))
    )  # antipodal directions encode same tensor
    return {
        "volume_count": nvol,
        "b0_count": int((~dw).sum()),
        "diffusion_count": int(dw.sum()),
        "max_FSL_MRtrix_angle_deg": float(angle.max()),
        "FSL_MRtrix_consistent": bool(angle.max() <= options["gradient_angle_tolerance_deg"]),
        "row_identity_scope": "Derivative volume index only; raw-volume rotation lineage remains pending",
        "sources_match": True,
        "anatomical_gradient_truth": "not_established",
    }, world


def design_condition(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    x, y, z = v.T
    s = np.linalg.svd(
        np.c_[x * x, y * y, z * z, np.sqrt(2) * x * y, np.sqrt(2) * x * z, np.sqrt(2) * y * z],
        compute_uv=False,
    )
    return float(s[0] / s[-1]) if len(s) == 6 and s[-1] > 1e-12 else np.inf


def coverage(original, selected):
    a = original / np.linalg.norm(original, axis=1, keepdims=True)
    b = selected / np.linalg.norm(selected, axis=1, keepdims=True)
    nearest = np.degrees(np.arccos(np.clip(np.max(abs(a @ b.T), axis=1), 0, 1)))
    return float(np.quantile(nearest, 0.95))


def motion_plan(b, world, fd, opt, identity_verified=False, tracking_verified=False):
    b = np.asarray(b, float)
    g = np.asarray(world, float)
    fd = np.asarray(fd, float)
    if len(fd) != len(b) or g.shape != (len(b), 3):
        raise ValueError("Motion/gradient layout mismatch")
    if opt["maximum_conditions"] != 3:
        raise ValueError("Only three predefined conditions allowed")
    dw = b > opt["b0_max"]
    finite = np.isfinite(fd) & (fd >= 0)
    low = finite & (fd <= opt["low_motion_fd_mm"])
    shell = np.rint(b / opt["shell_width"]).astype(int)
    lowids = np.flatnonzero(dw & low)
    b0ids = np.flatnonzero(~dw & low)
    reasons = []
    control = []
    angles = []
    if len(lowids) < opt["minimum_directions"]:
        reasons.append("insufficient_low_motion_directions")
    if len(b0ids) < opt["minimum_b0"]:
        reasons.append("insufficient_low_motion_b0")
    if len(lowids) < opt["minimum_retained_fraction"] * int(dw.sum()):
        reasons.append("retained_fraction")
    # Preserve shell-bin counts and acquisition-order quartiles; no random cherry picking.
    time = np.minimum(np.arange(len(b)) * 4 // len(b), 3)
    shell_coverage = {}
    for s in np.unique(shell[dw]):
        original = dw & (shell == s)
        selected = original & low
        if selected.sum() < opt["minimum_retained_fraction"] * original.sum():
            reasons.append("lost_radial_coverage")
        if selected.any():
            shell_coverage[int(s)] = coverage(g[original], g[selected])
            if shell_coverage[int(s)] > opt["coverage_angle_p95_deg"]:
                reasons.append("shell_angular_coverage")
        else:
            reasons.append("lost_radial_bin")
    for s, t in sorted(set(zip(shell[lowids], time[lowids]))):
        ids = lowids[(shell[lowids] == s) & (time[lowids] == t)]
        candidates = np.flatnonzero(dw & finite & (shell == s) & (time == t))
        if len(candidates) < len(ids):
            reasons.append("unmatchable_stratum")
            continue
        a = g[ids] / np.linalg.norm(g[ids], axis=1, keepdims=True)
        v = g[candidates] / np.linalg.norm(g[candidates], axis=1, keepdims=True)
        distance = np.degrees(np.arccos(np.clip(abs(a @ v.T), 0, 1)))
        # Favor higher-motion candidates only within the predeclared angular tolerance.
        motion = fd[candidates] / max(float(fd[candidates].max()), 1)
        cost = np.where(distance <= opt["maximum_matching_angle_deg"], distance / 100 - motion, 1e6)
        ii, jj = linear_sum_assignment(cost)
        if len(ii) != len(ids) or (cost[ii, jj] >= 1e6).any():
            reasons.append("angular_matching_failed")
        control.extend(candidates[jj].tolist())
        angles.extend(distance[ii, jj].tolist())
    diagnostics = {
        "shell_bin_coverage_p95_deg": shell_coverage,
        "unknown_motion_volumes": int((~finite).sum()),
        "candidate_low_directions": len(lowids),
        "candidate_b0": len(b0ids),
        "angle_max_deg": max(angles) if angles else None,
    }
    if len(lowids) >= 6 and len(control) == len(lowids):
        diagnostics.update(
            coverage_p95_deg=coverage(g[dw], g[lowids]),
            design_condition=design_condition(g[lowids]),
            control_design_condition=design_condition(g[control]),
            motion_mean_difference_mm=float(fd[control].mean() - fd[lowids].mean()),
        )
        if diagnostics["coverage_p95_deg"] > opt["coverage_angle_p95_deg"]:
            reasons.append("angular_coverage")
        if (
            max(diagnostics["design_condition"], diagnostics["control_design_condition"])
            > opt["maximum_design_condition"]
        ):
            reasons.append("ill_conditioned_direction_design")
        if diagnostics["motion_mean_difference_mm"] < opt["matching_motion_difference_mm"]:
            reasons.append("insufficient_motion_contrast")
    else:
        reasons.append("insufficient_matched_design")
    computational_feasible = not reasons
    if not identity_verified:
        reasons.append("raw_volume_identity_unverified")
    if not tracking_verified:
        reasons.append("tracking_recipe_seed_unverified")
    conditions = [{"name": "existing_all_volume_GQI", "reuse": True, "reconstruction": False}]
    if not reasons:
        conditions += [
            {
                "name": "low_motion_matched",
                "indices": sorted([*lowids.tolist(), *b0ids.tolist()]),
                "reconstruction": "requires_separate_approval",
            },
            {
                "name": "matched_count_qspace_control",
                "indices": sorted([*control, *b0ids.tolist()]),
                "reconstruction": "requires_separate_approval",
            },
        ]
    return {
        "status": "feasible_requires_approval" if not reasons else "blocked",
        "computational_sampling_feasible": computational_feasible,
        "reasons": sorted(set(reasons)),
        "diagnostics": diagnostics,
        "conditions": conditions,
        "maximum_total_conditions": 3,
        "no_automatic_threshold_relaxation": True,
        "scope": "Sensitivity of existing corrected DWI; cannot undo upstream motion/model bias",
    }


def compare_connectivity(end, passed, end_length, pass_length, ids, expected_ids):
    e, p, el, pl = map(matrix, [end, passed, end_length, pass_length])
    ids = np.asarray(ids).reshape(-1)
    expected = np.asarray(expected_ids).reshape(-1)
    if (
        len(np.unique(ids)) != len(ids)
        or not np.array_equal(ids, expected)
        or e.shape != p.shape
        or e.shape != (len(ids), len(ids))
    ):
        raise ValueError("Atlas ordering/layout mismatch")
    if el.shape != e.shape or pl.shape != e.shape:
        raise ValueError("Length layout mismatch")
    iu = np.triu_indices(len(ids), 1)
    a = e[iu] > 0
    c = p[iu] > 0
    both = a & c
    rows = []
    for i, j in zip(*iu):
        rows.append(
            {
                "atlas_i": int(ids[i]),
                "atlas_j": int(ids[j]),
                "end_count": float(e[i, j]),
                "pass_count": float(p[i, j]),
                "end_present": bool(e[i, j] > 0),
                "pass_present": bool(p[i, j] > 0),
                "end_length_mm": float(el[i, j]) if e[i, j] > 0 else None,
                "pass_length_mm": float(pl[i, j]) if p[i, j] > 0 else None,
            }
        )
    warnings = []
    if np.any((e > 0) & (el <= 0)) or np.any((p > 0) & (pl <= 0)):
        warnings.append("present_edge_nonpositive_length")
    summary = {
        "end_edges": int(a.sum()),
        "pass_edges": int(c.sum()),
        "intersection": int(both.sum()),
        "jaccard": float(both.sum() / (a | c).sum()) if (a | c).any() else None,
        "end_only": int((a & ~c).sum()),
        "pass_only": int((c & ~a).sum()),
        "warnings": warnings,
        "false_positive_or_negative_rate": "not_identifiable",
        "interpretation": "Connection-definition sensitivity; shared streamlines are not independent replicates",
    }
    return summary, rows


def edge_uncertainty(weights, lengths, ids_by_condition, reference_scale):
    if (
        not 2 <= len(weights) <= 3
        or len(weights) != len(lengths)
        or len(weights) != len(ids_by_condition)
        or not np.isfinite(reference_scale)
        or reference_scale <= 0
    ):
        raise ValueError("At least two aligned conditions and a positive common scale required")
    if any(not np.array_equal(ids_by_condition[0], x) for x in ids_by_condition[1:]):
        raise ValueError("Order mismatch")
    w = np.stack([matrix(x) for x in weights])
    l = np.stack([matrix(x) for x in lengths])
    if w.shape != l.shape:
        raise ValueError("Length/weight shape mismatch")
    present = w > 0
    if np.any(l[present] <= 0):
        raise ValueError("Present edge has no valid length")
    # Missing edges are zero weight but have no tract length; never invent zero lengths.
    lengths_masked = np.where(present, l, np.nan)
    rows = []
    for i, j in zip(*np.triu_indices(w.shape[1], 1)):
        values = w[:, i, j] / reference_scale
        v = lengths_masked[:, i, j]
        finite = v[np.isfinite(v)]
        rows.append(
            {
                "i": i,
                "j": j,
                "presence_fraction": float(present[:, i, j].mean()),
                "weight_mean": float(values.mean()),
                "weight_range": float(np.ptp(values)),
                "weight_cv": float(values.std(ddof=1) / values.mean())
                if values.mean() > 0
                else None,
                "length_min_mm": float(finite.min()) if len(finite) else None,
                "length_max_mm": float(finite.max()) if len(finite) else None,
                "length_observations": len(finite),
            }
        )
    return {
        "edge_rows": rows,
        "uncertainty_type": "descriptive condition sensitivity",
        "confidence_intervals": "not_estimated; <=3 nonindependent conditions are not a sampling distribution",
    }
