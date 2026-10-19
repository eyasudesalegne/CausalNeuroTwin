"""Source selection uncertainty, not connectivity changes or dipole calibration."""

import numpy as np

from .atlas import support


def registration_gate(evidence, leadfield_sha256):
    if (
        not evidence
        or evidence.get("decision") != "ACCEPT"
        or evidence.get("independent_review") is not True
    ):
        raise RuntimeError("Verified electrode/MRI registration evidence required")
    if (
        evidence.get("leadfield_sha256") != leadfield_sha256
        or not evidence.get("reviewer")
        or not evidence.get("evidence")
    ):
        raise RuntimeError("Registration does not attest this exact forward asset")
    if (
        evidence.get("reference") != "average"
        or not evidence.get("channel_order_verified")
        or not evidence.get("source_order_verified")
    ):
        raise RuntimeError("Forward reference/channel/source order unverified")


def aggregate(
    index, vertex_gain, labels, mask, areas, baseline_areas, policy="fixed_baseline_area"
):
    Gv = np.asarray(vertex_gain, float)
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    areas = np.asarray(areas, float)
    if (
        Gv.ndim != 2
        or Gv.shape[1] != len(labels)
        or mask.shape != labels.shape
        or mask.dtype != bool
        or areas.shape != labels.shape
    ):
        raise ValueError("Leadfield/source mask layout")
    if not np.isfinite(Gv).all() or not np.isfinite(areas).all() or (areas < 0).any():
        raise ValueError("Nonfinite gain or invalid areas")
    if policy not in ("fixed_baseline_area", "retained_area"):
        raise ValueError("Explicit normalization policy required")
    rows, retained = support(index, labels, mask, areas, baseline_areas)
    denominator = np.asarray(baseline_areas if policy == "fixed_baseline_area" else retained, float)
    if (
        denominator.shape != (index.n,)
        or (denominator < 0).any()
        or not np.isfinite(denominator).all()
    ):
        raise ValueError("Parcel denominators")
    out = np.zeros((Gv.shape[0], index.n))
    mass = np.zeros(index.n)
    for j, aid in enumerate(index.ids):
        selected = mask & (labels == aid)
        if aid <= 210 and selected.any():
            if denominator[j] <= 0:
                raise ValueError("Nonempty parcel has no normalization area")
            w = areas[selected] / denominator[j]
            out[:, j] = Gv[:, selected] @ w
            mass[j] = w.sum()
    return out, {
        "normalization": policy,
        "column_mass": mass.tolist(),
        "support": rows,
        "retained_area_renormalization_changes_source_gain": policy == "retained_area",
        "noncortical_nodes_remain_in_dynamics": True,
        "JR_to_dipole_amplitude_calibrated": False,
    }


def unresolved_bracket(index, Gv, labels, core, unresolved, area, denominator, alpha):
    if not 0 <= alpha <= 1 or np.any(core & unresolved):
        raise ValueError("Disjoint components and alpha in [0,1]")
    a, _ = aggregate(index, Gv, labels, core, area, denominator)
    b, _ = aggregate(index, Gv, labels, unresolved, area, denominator)
    return a + alpha * b


def operator_difference(reference, candidate):
    a = np.asarray(reference, float)
    b = np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparable finite operators required")
    denom = np.linalg.norm(a, axis=0)
    relative = np.divide(
        np.linalg.norm(b - a, axis=0), denom, out=np.full(len(denom), np.nan), where=denom > 0
    )
    return {
        "relative_frobenius": float(np.linalg.norm(b - a) / np.linalg.norm(a))
        if np.linalg.norm(a) > 0
        else None,
        "column_relative_difference": relative.tolist(),
        "no_accepted_anatomical_accuracy_inference": True,
    }
