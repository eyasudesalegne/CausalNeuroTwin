"""Reuse stored assignments and stored geodesic distances; no propagation."""

import numpy as np


def parcel_masks(row, direct, final, distance, nleft):
    aid = int(row["atlas_id"])
    correct = (
        np.arange(len(direct)) < nleft
        if row["label"].endswith("_L")
        else np.arange(len(direct)) >= nleft
        if row["label"].endswith("_R")
        else None
    )
    if correct is None:
        raise ValueError("Parcel hemisphere suffix absent")
    seeds = (direct == aid) & correct
    rejected = (direct == aid) & ~correct
    added = (final == aid) & ~seeds
    if not seeds.any() or np.any((final == aid) & ~correct):
        raise ValueError("Missing seeds or final hemisphere violation")
    if (
        np.any(final[seeds] != aid)
        or not np.isfinite(distance[final == aid]).all()
        or np.any(distance[final == aid] < 0)
    ):
        raise ValueError("Stored completion inconsistent with original seeds/distances")
    if (
        int(seeds.sum()) != int(row["seed_vertices"])
        or int((final == aid).sum()) != int(row["final_vertices"])
        or int(rejected.sum()) != int(row["wrong_hemisphere_samples"])
    ):
        raise ValueError("Saved queue and label arrays disagree")
    if not np.allclose(distance[seeds], 0, atol=1e-6):
        raise ValueError("Stored trusted-seed distances are not zero")
    if int((distance[added] > 10).sum()) != int(row["added_gt10mm"]):
        raise ValueError("Saved long-range completion count changed")
    summary = {
        "atlas_id": aid,
        "label": row["label"],
        "seed_vertices": int(seeds.sum()),
        "historical_rejected_wrong_hemisphere": int(rejected.sum()),
        "unaccepted_geodesic_additions": int(added.sum()),
        "added_distance_p50_mm": float(np.median(distance[added])) if added.any() else None,
        "added_distance_max_mm": float(distance[added].max()) if added.any() else None,
        "added_gt10mm": int((distance[added] > 10).sum()),
        "added_gt20mm": int((distance[added] > 20).sum()),
        "trusted_fraction_is_accuracy": False,
        "assessment_status": "pending",
    }
    return seeds, rejected, added, summary


def load_arrays(old, spec, nvertices):
    direct = np.load(old["artifacts"]["direct_labels"]["path"], allow_pickle=False)
    final = np.load(old["artifacts"]["geodesic_labels"]["path"], allow_pickle=False)
    distance = np.load(spec["path"], allow_pickle=False)
    if any(a.shape != (nvertices,) for a in (direct, final, distance)):
        raise ValueError("Stored per-vertex array layout mismatch")
    if (
        str(distance.dtype) != spec["expected_dtype"]
        or list(distance.shape) != spec["expected_shape"]
    ):
        raise ValueError("Saved distance header drift")
    if not np.issubdtype(direct.dtype, np.integer) or not np.issubdtype(final.dtype, np.integer):
        raise ValueError("Stored labels must be categorical integers")
    return direct, final, distance
