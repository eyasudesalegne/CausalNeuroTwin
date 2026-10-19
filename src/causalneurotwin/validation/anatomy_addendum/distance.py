"""Exact distances to axis-orthogonal voxel-cell interfaces, in physical millimeters.

KD-tree prunes faces, but distances are to finite rectangles, not face centers.
Reject shear rather than apply an invalid Euclidean distance metric.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def apply(matrix, points):
    from causalneurotwin.validation.anatomy.geometry import apply as verified_apply

    return verified_apply(matrix, points)


class VoxelBoundary:
    def __init__(self, labels, affine, region=None, pairs=None):
        self.affine = np.asarray(affine, float)
        self.spacing = np.linalg.norm(self.affine[:3, :3], axis=0)
        basis = self.affine[:3, :3] / self.spacing
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("Sheared affine: voxel-face distance requires orthogonal axes")
        if (region is None) == (pairs is None):
            raise ValueError("Supply a region OR interface pairs")
        a = np.asarray(labels)
        centers, axes = [], []
        for axis in range(3):
            left = [slice(None)] * 3
            right = left.copy()
            left[axis], right[axis] = slice(0, -1), slice(1, None)
            lo, hi = a[tuple(left)], a[tuple(right)]
            if region is not None:
                select = (lo == region) != (hi == region)
            else:
                select = np.zeros(lo.shape, bool)
                for x, y in pairs:
                    select |= ((lo == x) & (hi == y)) | ((lo == y) & (hi == x))
            ijk = np.argwhere(select).astype(float)
            ijk[:, axis] += 0.5
            centers.append(ijk * self.spacing)
            axes.append(np.full(len(ijk), axis, np.int8))
            if region is not None:
                for side in (0, a.shape[axis] - 1):
                    edge = np.take(a, side, axis=axis) == region
                    coords = np.argwhere(edge).astype(float)
                    xyz = np.insert(coords, axis, -0.5 if side == 0 else side + 0.5, axis=1)
                    centers.append(xyz * self.spacing)
                    axes.append(np.full(len(xyz), axis, np.int8))
        self.centers = np.concatenate(centers)
        self.axes = np.concatenate(axes)
        self.tree = cKDTree(self.centers) if len(self.centers) else None
        self.radius = max(np.linalg.norm(np.delete(self.spacing, axis)) / 2 for axis in range(3))

    def query(self, ras, chunk=2048):
        points = np.asarray(ras, float).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite distance query")
        result = np.full(len(points), np.nan)
        if self.tree is None:
            return result
        q = apply(np.linalg.inv(self.affine), points) * self.spacing
        for start in range(0, len(q), chunk):
            x = q[start : start + chunk]
            nearest, _ = self.tree.query(x)
            # Any face farther than nearest-center + largest half diagonal cannot win.
            groups = self.tree.query_ball_point(x, nearest + self.radius + 1e-10)
            lengths = np.array([len(g) for g in groups])
            ids = np.concatenate(groups).astype(np.int64)
            owner = np.repeat(np.arange(len(x)), lengths)
            half = np.tile(self.spacing / 2, (len(ids), 1))
            half[np.arange(len(ids)), self.axes[ids]] = 0
            delta = np.maximum(np.abs(x[owner] - self.centers[ids]) - half, 0)
            d = np.linalg.norm(delta, axis=1)
            best = np.full(len(x), np.inf)
            np.minimum.at(best, owner, d)
            result[start : start + len(x)] = best
        return result


def surface_geometry(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    weights = np.zeros(len(vertices))
    normals = np.zeros_like(vertices, dtype=float)
    for j in range(3):
        np.add.at(weights, faces[:, j], areas / 3)
        np.add.at(normals, faces[:, j], cross)
    size = np.linalg.norm(normals, axis=1)
    valid = size > 1e-12
    normals[valid] /= size[valid, None]
    normals[~valid] = np.nan
    return (
        weights,
        normals,
        {
            "degenerate_faces": int((areas <= 1e-12).sum()),
            "undefined_vertex_normals": int((~valid).sum()),
        },
    )


def clusters(vertices, faces, mask, weights, distances):
    ids = np.flatnonzero(mask)
    labels = np.full(len(vertices), -1, int)
    if not len(ids):
        return labels, []
    inverse = np.full(len(vertices), -1, int)
    inverse[ids] = np.arange(len(ids))
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges[mask[edges].all(axis=1)]
    graph = coo_matrix(
        (np.ones(len(edges)), (inverse[edges[:, 0]], inverse[edges[:, 1]])),
        shape=(len(ids), len(ids)),
    ).tocsr()
    _n, component = connected_components(graph, directed=False)
    rows = []
    order = np.argsort(component)
    chunks = np.split(ids[order], np.flatnonzero(np.diff(component[order])) + 1)
    for group in chunks:
        center = (
            np.average(vertices[group], axis=0, weights=weights[group])
            if weights[group].sum()
            else vertices[group].mean(axis=0)
        )
        finite = distances[group][np.isfinite(distances[group])]
        rows.append(
            {
                "ids": group,
                "vertices": len(group),
                "area_mm2": float(weights[group].sum()),
                "centroid_ras_mm": center.tolist(),
                "bbox_min_ras_mm": vertices[group].min(0).tolist(),
                "bbox_max_ras_mm": vertices[group].max(0).tolist(),
                "bbox_diagonal_mm": float(np.linalg.norm(np.ptp(vertices[group], axis=0))),
                "max_boundary_distance_mm": float(finite.max()) if len(finite) else None,
            }
        )
    rows.sort(key=lambda r: -r["area_mm2"])
    for number, row in enumerate(rows):
        labels[row.pop("ids")] = number
        row["cluster_id"] = number
    return labels, rows


def analyze_surface(volume, vertices, faces, boundary, offsets):
    offsets = np.asarray(offsets, float)
    if not len(offsets) or not np.isfinite(offsets).all() or 0 not in offsets:
        raise ValueError("Finite normal offsets including zero required")
    if not np.allclose(np.sort(offsets), -np.sort(offsets)[::-1]):
        raise ValueError(
            "Normal offsets must be symmetric; winding is not inward/outward certified"
        )
    tissue = volume.sample(vertices, order=0)
    disagreement = ~np.isfinite(tissue) | (tissue != 2)
    distance = boundary.query(vertices)
    inside = np.isfinite(tissue) & (tissue == 2)
    signed = np.where(inside, -distance, distance)
    weights, normals, geometry = surface_geometry(vertices, faces)
    normal_ok = np.isfinite(normals).all(axis=1)
    sampled = np.full((len(vertices), len(offsets)), np.nan)
    for i, offset in enumerate(offsets):
        sampled[normal_ok, i] = volume.sample(
            vertices[normal_ok] + offset * normals[normal_ok], order=0
        )
    complete = np.isfinite(sampled).all(axis=1)
    persistent = disagreement & complete & (sampled != 2).all(axis=1)
    unresolved = disagreement & ~complete
    cluster_id, rows = clusters(vertices, faces, disagreement, weights, distance)
    persistent_counts = np.bincount(cluster_id[persistent], minlength=len(rows))
    persistent_areas = np.bincount(
        cluster_id[persistent], weights=weights[persistent], minlength=len(rows)
    )
    for row in rows:
        row["persistent_vertices"] = int(persistent_counts[row["cluster_id"]])
        row["persistent_area_mm2"] = float(persistent_areas[row["cluster_id"]])
    categories = {}
    for label in np.unique(tissue[np.isfinite(tissue)]):
        mask = tissue == label
        categories[str(int(label))] = {
            "vertices": int(mask.sum()),
            "area_mm2": float(weights[mask].sum()),
            "boundary_distance_p50_mm": float(np.median(distance[mask])),
            "boundary_distance_p95_mm": float(np.quantile(distance[mask], 0.95)),
        }
    categories["outside_or_nonfinite"] = {
        "vertices": int((~np.isfinite(tissue)).sum()),
        "area_mm2": float(weights[~np.isfinite(tissue)].sum()),
    }
    stats = {
        "vertices": len(vertices),
        "disagreement_vertices": int(disagreement.sum()),
        "surface_area_mm2": float(weights.sum()),
        "disagreement_area_mm2": float(weights[disagreement].sum()),
        "persistent_disagreement_vertices": int(persistent.sum()),
        "persistent_disagreement_area_mm2": float(weights[persistent].sum()),
        "offset_indeterminate_vertices": int(unresolved.sum()),
        "offset_sensitive_disagreement_vertices": int(
            (disagreement & complete & ~persistent).sum()
        ),
        "categories": categories,
        "geometry": geometry,
        "clusters": rows,
        "offsets_mm": offsets.tolist(),
        "boundary_faces": len(boundary.centers),
        "boundary_distance_quantiles_mm": np.quantile(
            distance[np.isfinite(distance)], [0, 0.5, 0.95, 1]
        ).tolist(),
        "area_method": "Barycentric vertex areas: one third of incident triangle area",
        "extent_method": "RAS bounding-box diagonal; not geodesic diameter",
        "distance_method": "Exact Euclidean distance to boundary of union of label-2 voxel cells; orthogonal affine",
        "sign": "Signed boundary distance negative inside label 2; region distance zero inside",
        "evidence_category": "computational_consistency_verified",
        "independent_anatomical_assessment": "pending",
    }
    arrays = {
        "tissue": tissue,
        "boundary_distance_mm": distance,
        "signed_boundary_distance_mm": signed,
        "gm_region_distance_mm": np.where(inside, 0, distance),
        "vertex_area_mm2": weights,
        "normal_offsets_tissue": sampled,
        "disagreement": disagreement,
        "persistent": persistent,
        "offset_indeterminate": unresolved,
        "cluster_id": cluster_id,
    }
    return stats, arrays
