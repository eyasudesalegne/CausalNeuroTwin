"""Direct seeds, rejected samples and unaccepted completion stay separate."""

import numpy as np

from .anatomy import reviewed


def mapping_evidence(index, direct, completed, distance, hemisphere, flagged):
    direct = np.asarray(direct)
    completed = np.asarray(completed)
    distance = np.asarray(distance)
    correct = index.correct_hemisphere(direct, hemisphere)
    if (
        direct.shape != completed.shape
        or distance.shape != direct.shape
        or not np.issubdtype(direct.dtype, np.integer)
    ):
        raise ValueError("Categorical source mapping layout")
    trusted = correct
    completed_correct = index.correct_hemisphere(completed, hemisphere)
    if not completed_correct.all() or not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Saved completion invalid")
    if np.any(completed[trusted] != direct[trusted]) or np.any(distance[trusted] != 0):
        raise ValueError("Trusted seeds altered")
    rows = []
    for item in flagged:
        aid = int(item["atlas_id"])
        seed = trusted & (direct == aid)
        if aid not in index.ids or aid > 210:
            raise ValueError("Flagged parcel outside cortical model")
        if int(seed.sum()) != int(item["seed_vertices"]):
            raise ValueError("Saved trusted-support count mismatch")
        rejected = (direct == aid) & ~correct
        added = (completed == aid) & ~seed
        rows.append(
            {
                "atlas_id": aid,
                "label": item["label"],
                "trusted_seed_vertices": int(seed.sum()),
                "historical_rejected_wrong_hemisphere": int(rejected.sum()),
                "unaccepted_completion_vertices": int(added.sum()),
                "completion_max_mm": float(distance[added].max()) if added.any() else None,
            }
        )
    return trusted, rows


def alternatives(index, direct, hemisphere, conservative, flagged_ids, reassignments=()):
    direct = np.asarray(direct)
    trusted = index.correct_hemisphere(direct, hemisphere)
    if np.shape(conservative) != trusted.shape or np.any(conservative & ~trusted):
        raise ValueError("Conservative selection adds untrusted seeds")
    labels = direct.copy()
    for row in reassignments:
        if not reviewed(row):
            raise ValueError("Alternative assignment needs independent evidence")
        v = row["vertex"]
        old = row["from_id"]
        new = row["to_id"]
        if (
            not isinstance(v, int)
            or not 0 <= v < len(labels)
            or labels[v] != old
            or not conservative[v]
        ):
            raise ValueError("Assignment does not target a retained original source")
        if new not in index.ids or new > 210:
            raise ValueError("Alternative label is not cortical")
        probe = np.array([new])
        side = np.array([hemisphere[v]])
        if not index.correct_hemisphere(probe, side)[0]:
            raise ValueError("Wrong-hemisphere alternative")
        labels[v] = new
    uncertain = np.isin(direct, list(flagged_ids)) & trusted
    return {
        "original_direct": (direct.copy(), trusted),
        "reviewed_conservative": (direct.copy(), np.asarray(conservative).copy()),
        "flagged_parcel_ablation": (direct.copy(), conservative & ~uncertain),
        "reviewed_alternative": (labels, np.asarray(conservative).copy())
        if reassignments
        else None,
    }


def support(index, labels, mask, areas, baseline_areas=None):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    areas = np.asarray(areas, float)
    if labels.shape != mask.shape or mask.shape != areas.shape or mask.dtype != bool:
        raise ValueError("Mapping/source layout")
    if not np.isfinite(areas).all() or (areas < 0).any():
        raise ValueError("Invalid saved source areas")
    if not np.isin(labels[mask], [x for x in index.ids if x <= 210]).all():
        raise ValueError("Selected source has unknown or noncortical label")
    if baseline_areas is not None and (
        np.shape(baseline_areas) != (index.n,)
        or not np.isfinite(baseline_areas).all()
        or np.any(np.asarray(baseline_areas) < 0)
    ):
        raise ValueError("Baseline area layout")
    rows = []
    denom = np.zeros(index.n)
    for j, (aid, name) in enumerate(zip(index.ids, index.names)):
        m = mask & (labels == aid)
        denom[j] = areas[m].sum() if aid <= 210 else 0
        if aid > 210 and m.any():
            raise ValueError("Noncortical cortical-source assignment")
        rows.append(
            {
                "tvb_index": j,
                "atlas_id": aid,
                "label": name,
                "vertices": int(m.sum()),
                "area_mm2": float(denom[j]),
                "supported": bool(denom[j] > 0),
                "retained_baseline_area_fraction": float(denom[j] / baseline_areas[j])
                if baseline_areas is not None and baseline_areas[j] > 0
                else None,
            }
        )
    return rows, denom
