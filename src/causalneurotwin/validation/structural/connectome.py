"""Existing network-definition variants; no reconstruction or tractography."""

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path


def validate(index, weights, lengths, ids):
    index.check_order(ids)
    index.check_square(weights)
    index.check_square(lengths)
    W = np.asarray(weights, float)
    L = np.asarray(lengths, float)
    if not np.isfinite(W).all() or not np.isfinite(L).all() or (W < 0).any() or (L < 0).any():
        raise ValueError("Finite nonnegative matrices required")
    if not np.allclose(W, W.T, atol=1e-12, rtol=0) or not np.allclose(L, L.T, atol=1e-9, rtol=0):
        raise ValueError("Asymmetric undirected matrices")
    if np.any(np.diag(W) != 0) or np.any(np.diag(L) != 0) or np.any(L[W > 0] <= 0):
        raise ValueError("Self edges or present edges without positive lengths")
    # Disconnected pairs have no physical length; require zero sentinel in serialized matrices.
    if np.any(L[W == 0] != 0):
        raise ValueError("Absent edges carry undocumented lengths")
    return W.copy(), L.copy()


def from_saved_edges(index, rows):
    n = index.n
    position = {x: i for i, x in enumerate(index.ids)}
    seen = set()
    end = np.zeros((n, n))
    passed = end.copy()
    el = end.copy()
    pl = end.copy()
    for row in rows:
        a, b = int(row["atlas_i"]), int(row["atlas_j"])
        if a not in position or b not in position or a == b:
            raise ValueError("Unknown edge ID")
        i, j = position[a], position[b]
        key = tuple(sorted((i, j)))
        if key in seen:
            raise ValueError("Duplicated saved edge")
        seen.add(key)
        for keyW, keyL, W, L in [
            ("end_count", "end_length_mm", end, el),
            ("pass_count", "pass_length_mm", passed, pl),
        ]:
            count = float(row[keyW])
            length = float(row[keyL]) if row[keyL] not in ("", None) else 0.0
            W[i, j] = W[j, i] = count
            L[i, j] = L[j, i] = length
    if len(seen) != n * (n - 1) // 2:
        raise ValueError("Incomplete saved all-pairs table")
    validate(index, end, el, index.ids)
    validate(index, passed, pl, index.ids)
    return end, passed, el, pl


def variants(index, end, passed, end_lengths, pass_lengths, reference_scale, nulls=()):
    e, el = validate(index, end, end_lengths, index.ids)
    p, pl = validate(index, passed, pass_lengths, index.ids)
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Positive reference scale")
    actual = float(np.abs(p).sum(1).max())
    if not np.isclose(actual, reference_scale, atol=1e-8, rtol=1e-12):
        raise ValueError("Normalization differs from original pass factor")
    common = (e > 0) & (p > 0)
    result = {
        "pass": {"weights": p / reference_scale, "lengths": pl, "role": "original pass definition"},
        "endpoint": {
            "weights": e / reference_scale,
            "lengths": el,
            "role": "endpoint definition, not superior ground truth",
        },
        "common_pass_weighted": {
            "weights": np.where(common, p, 0) / reference_scale,
            "lengths": np.where(common, pl, 0),
            "role": "common-edge support with pass weights/lengths; not accepted consensus truth",
        },
    }
    if len(nulls) > 20:
        raise ValueError("Existing ensemble bounded to 20 members")
    ref = p / reference_scale
    for null in nulls:
        key = null["id"]
        if not key.startswith("null_") or key in result:
            raise ValueError("Null identity")
        W, L = validate(index, null["weights"], null["lengths"], null["ids"])
        if null.get("units") != "original_pass_region_scaled":
            raise ValueError("Null units ambiguous; refuse double normalization")
        if not np.array_equal((W > 0).sum(1), (p > 0).sum(1)):
            raise ValueError("Null degree claim fails")
        target = ref.sum(1)
        if np.max(abs(W.sum(1) - target) / np.maximum(target, 1e-15)) > 1e-10:
            raise ValueError("Null strength claim fails")
        iu = np.triu_indices(index.n, 1)
        if not np.array_equal(np.sort(L[iu][W[iu] > 0]), np.sort(pl[iu][p[iu] > 0])):
            raise ValueError("Null global length distribution claim fails")
        result[key] = {
            "weights": W,
            "lengths": L,
            "role": "existing structural-organization null; weights/length pairings changed",
        }
    return result


def distribution(x):
    x = np.asarray(x, float)
    return (
        {
            "n": len(x),
            "min": float(x.min()),
            "median": float(np.median(x)),
            "p95": float(np.quantile(x, 0.95)),
            "max": float(x.max()),
        }
        if len(x)
        else {"n": 0}
    )


def network_metrics(index, variant, speed=4.0):
    W, L = validate(index, variant["weights"], variant["lengths"], index.ids)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Positive speed in mm/ms required")
    A = W > 0
    n = index.n
    iu = np.triu_indices(n, 1)
    present = A[iu]
    degree = A.sum(1)
    strength = W.sum(1)
    d = shortest_path(A.astype(float), directed=False, unweighted=True)
    inv = np.divide(1, d, out=np.zeros_like(d), where=(d > 0) & np.isfinite(d))
    clustering = []
    for i in range(n):
        neighbors = np.flatnonzero(A[i])
        k = len(neighbors)
        clustering.append(
            float(A[np.ix_(neighbors, neighbors)].sum() / (k * (k - 1))) if k > 1 else 0.0
        )
    return {
        "edges": int(present.sum()),
        "density": float(2 * present.sum() / (n * (n - 1))),
        "components": int(connected_components(A, directed=False, return_labels=False)),
        "degree": degree.tolist(),
        "strength": strength.tolist(),
        "binary_local_clustering": clustering,
        "binary_global_efficiency": float(inv.sum() / (n * (n - 1))),
        "weight_distribution": distribution(W[iu][present]),
        "length_mm_distribution": distribution(L[iu][present]),
        "delay_ms_distribution": distribution(L[iu][present] / speed),
        "speed_mm_ms": speed,
        "maximum_row_sum": float(strength.max()),
        "total_strength": float(strength.sum()),
        "interpretation": "descriptive network-definition/organization sensitivity, not acquisition uncertainty or edge truth",
    }


def compare(index, reference, candidate):
    a = reference["weights"] > 0
    b = candidate["weights"] > 0
    iu = np.triu_indices(index.n, 1)
    both = (a & b)[iu]
    union = (a | b)[iu]
    return {
        "shared": int(both.sum()),
        "reference_only": int((a & ~b)[iu].sum()),
        "candidate_only": int((b & ~a)[iu].sum()),
        "jaccard": float(both.sum() / union.sum()) if union.any() else None,
        "degree_delta": (b.sum(1) - a.sum(1)).tolist(),
        "strength_delta": (candidate["weights"].sum(1) - reference["weights"].sum(1)).tolist(),
        "normalization_policy": "single frozen pass-derived scale; no per-model retuning",
        "equal_total_strength_secondary_factor": float(
            reference["weights"].sum() / candidate["weights"].sum()
        )
        if candidate["weights"].sum() > 0
        else None,
        "secondary_factor_applied": False,
        "absent_edges_have_no_measured_length": True,
    }
