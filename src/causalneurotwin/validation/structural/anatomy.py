"""Review-based exclusions, template candidates and unresolved anatomy are distinct."""

import numpy as np


def reviewed(record):
    return (
        record.get("decision") == "ACCEPT"
        and record.get("independent") is True
        and bool(record.get("reviewer"))
        and bool(record.get("date"))
        and bool(record.get("evidence"))
        and len(record.get("evidence_sha256", "")) == 64
    )


def selections(trusted, disagreement, persistent, template_unknown, annotations=()):
    trusted = np.asarray(trusted)
    n = len(trusted)
    if trusted.dtype != bool or trusted.ndim != 1:
        raise ValueError("Boolean source mask required")
    values = [np.asarray(x) for x in (disagreement, persistent, template_unknown)]
    if any(x.shape != (n,) or x.dtype != bool for x in values):
        raise ValueError("Mask layout/dtype mismatch")
    if np.any(values[1] & ~values[0]):
        raise ValueError("Persistence outside disagreement")
    excluded = np.zeros(n, bool)
    resolved_cortical = np.zeros(n, bool)
    for row in annotations:
        if not reviewed(row):
            raise ValueError("Unsigned/non-independent exclusion cannot alter a source mask")
        vertex = row["vertex"]
        if not isinstance(vertex, int) or not 0 <= vertex < n:
            raise ValueError("Invalid reviewed vertex")
        kind = row["classification"]
        if kind in ("noncortical_closure", "invalid_source"):
            excluded[vertex] = True
        elif kind == "cortical":
            resolved_cortical[vertex] = True
        else:
            raise ValueError("Unsupported anatomical disposition")
    if np.any(excluded & resolved_cortical):
        raise ValueError("Conflicting anatomical dispositions")
    conservative = trusted & ~excluded
    unresolved = conservative & (values[0] | values[2]) & ~resolved_cortical
    masks = {
        "original_trusted": trusted.copy(),
        "reviewed_conservative": conservative,
        "sensitivity_core": conservative & ~unresolved,
        "unresolved_component": unresolved,
    }
    assert np.array_equal(masks["sensitivity_core"] | unresolved, conservative)
    return masks, {
        "independent_global_acceptance": False,
        "reviewed_excluded_vertices": int((trusted & excluded).sum()),
        "unresolved_vertices": int(unresolved.sum()),
        "template_unknown_is_ground_truth": False,
        "conservative_status": "reviewed_exclusions_applied"
        if excluded.any()
        else "no_reviewed_exclusions; identical_to_baseline",
        "sensitivity_core_interpretation": "hypothetical omission bound, not accepted anatomy",
    }


def geometry_diagnostics(points, area, masks, cluster_ids=None, hemisphere=None):
    points = np.asarray(points, float)
    area = np.asarray(area, float)
    n = len(area)
    if (
        points.shape != (n, 3)
        or not np.isfinite(points).all()
        or not np.isfinite(area).all()
        or (area < 0).any()
    ):
        raise ValueError("Finite RAS geometry and nonnegative saved areas required")
    out = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (n,):
            raise ValueError("Mask layout")
        total = float(area[mask].sum())
        out[name] = {
            "vertices": int(mask.sum()),
            "area_mm2": total,
            "retention_fraction": float(mask.sum() / max(1, masks["original_trusted"].sum())),
            "area_weighted_centroid_ras_mm": np.average(
                points[mask], axis=0, weights=area[mask]
            ).tolist()
            if total > 0
            else None,
            "bounds_ras_mm": [points[mask].min(0).tolist(), points[mask].max(0).tolist()]
            if mask.any()
            else None,
        }
    baseline = out["original_trusted"]
    for name, row in out.items():
        row["area_retention_fraction"] = (
            row["area_mm2"] / baseline["area_mm2"] if baseline["area_mm2"] > 0 else None
        )
        centers = (row["area_weighted_centroid_ras_mm"], baseline["area_weighted_centroid_ras_mm"])
        row["centroid_shift_from_baseline_mm"] = (
            float(np.linalg.norm(np.asarray(centers[0]) - centers[1]))
            if all(x is not None for x in centers)
            else None
        )
        if hemisphere is not None:
            h = np.asarray(hemisphere)
            if h.shape != (n,) or not np.isin(h, ["L", "R"]).all():
                raise ValueError("Explicit hemisphere layout")
            row["by_hemisphere"] = {
                side: {
                    "vertices": int((masks[name] & (h == side)).sum()),
                    "area_mm2": float(area[masks[name] & (h == side)].sum()),
                }
                for side in ["L", "R"]
            }
    return out


def cluster_overlap(cluster_ids, hemisphere, template_unknown, disagreement, persistent):
    ids = np.asarray(cluster_ids)
    hemi = np.asarray(hemisphere)
    rows = []
    if not all(
        np.shape(x) == ids.shape for x in [hemi, template_unknown, disagreement, persistent]
    ):
        raise ValueError("Cluster evidence alignment")
    for side in ["L", "R"]:
        for cid in np.unique(ids[(hemi == side) & (ids >= 0)]):
            m = (hemi == side) & (ids == cid)
            n = int(m.sum())
            rows.append(
                {
                    "hemisphere": side,
                    "saved_cluster_id": int(cid),
                    "vertices": n,
                    "template_unknown_overlap": int((m & template_unknown).sum()),
                    "persistent_disagreement": int((m & persistent).sum()),
                    "assessment": "indeterminate; template agreement is a closure-convention hypothesis, not anatomical acceptance",
                }
            )
    return rows


def template_candidates(subject_registered_sphere, template_sphere, template_mask):
    """Future compute-node operation, matching inspected SimNIBS resampling convention."""
    import nibabel as nib
    from cortech import Sphere

    mask = nib.freesurfer.io.read_morph_data(str(template_mask))
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Template mask is not binary")
    template = Sphere.from_file(str(template_sphere))
    subject = Sphere.from_file(str(subject_registered_sphere))
    transferred = template.project_and_resample(subject, mask, method="nearest")
    if not np.isin(transferred, [0, 1]).all():
        raise ValueError("Categorical transfer changed labels")
    return np.asarray(transferred) == 0
