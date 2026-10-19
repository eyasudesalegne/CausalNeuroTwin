"""New tissue threshold cross-tabs and spatial localization, not a repeated audit."""

from collections import Counter

import numpy as np

CRITERIA = {
    "mean_ratio_below": ("mean_ratio", "lt"),
    "scaled_jacobian_below": ("scaled_jacobian", "lt"),
    "min_dihedral_below_deg": ("min_dihedral_deg", "lt"),
    "max_dihedral_above_deg": ("max_dihedral_deg", "gt"),
}


def flags(quality, thresholds):
    return {
        name: quality[key] < thresholds[name]
        if direction == "lt"
        else quality[key] > thresholds[name]
        for name, (key, direction) in CRITERIA.items()
    }


def stream_localization(path, prior, interfaces, tissue_volume, options, retained_ids):
    if options["spatial_bin_mm"] <= 0:
        raise ValueError("Spatial bin must be positive")
    from causalneurotwin.validation.anatomy.mesh import tetra_quality

    thresholds = prior["triage_thresholds_provisional"]
    counts, totals, spatial, global_counts = {}, Counter(), Counter(), Counter()
    rows, geometry = [], {}
    seen = 0
    for ids, tags, xyz in GmshChunks(path):
        q = tetra_quality(xyz)
        masks = flags(q, thresholds)
        finite = np.isfinite(np.stack(list(q.values()))).all(axis=0)
        bad = np.logical_or.reduce(list(masks.values())) | ~finite
        centroids = xyz.mean(axis=1)
        for tag in np.unique(tags):
            select = tags == tag
            totals[int(tag)] += int(select.sum())
            counts.setdefault(int(tag), Counter())
            for name, mask in masks.items():
                counts[int(tag)][name] += int((mask & select).sum())
            counts[int(tag)]["nonfinite"] += int((~finite & select).sum())
            counts[int(tag)]["any_warning"] += int((bad & select).sum())
        global_counts.update({k: int(v.sum()) for k, v in masks.items()})
        bad_ids = np.flatnonzero(bad)
        positions = centroids[bad_ids]
        distances = {name: interface.query(positions) for name, interface in interfaces.items()}
        sampled = tissue_volume.sample(positions, order=0)
        for j, i in enumerate(bad_ids):
            reasons = ";".join(k for k, mask in masks.items() if mask[i])
            if not finite[i]:
                reasons += ";nonfinite"
            row = {
                "element_id": int(ids[i]),
                "tissue_tag": int(tags[i]),
                "x_mm": float(centroids[i, 0]),
                "y_mm": float(centroids[i, 1]),
                "z_mm": float(centroids[i, 2]),
                "criteria": reasons,
                "centroid_image_label": float(sampled[j]) if np.isfinite(sampled[j]) else None,
                **{k: float(v[i]) for k, v in q.items()},
                **{
                    name + "_distance_mm": float(d[j]) if np.isfinite(d[j]) else None
                    for name, d in distances.items()
                },
            }
            rows.append(row)
            location = tuple(np.floor(centroids[i] / options["spatial_bin_mm"]).astype(int))
            spatial[(int(tags[i]), *location)] += 1
        for i, eid in enumerate(ids):
            if int(eid) in retained_ids:
                geometry[int(eid)] = {
                    "vertices_ras_mm": xyz[i].tolist(),
                    "tissue_tag": int(tags[i]),
                }
        seen += len(ids)
    if seen != prior["tetrahedra_evaluated"] or dict(global_counts) != prior["triage_counts"]:
        raise ValueError("Streamed classification differs from sealed v1 counts")
    if set(geometry) != set(retained_ids):
        raise ValueError("Saved representative element ID missing")
    per_tissue = []
    for tag, total in sorted(totals.items()):
        for criterion, count in sorted(counts[tag].items()):
            per_tissue.append(
                {
                    "tissue_tag": tag,
                    "criterion": criterion,
                    "count": count,
                    "total_tetrahedra": total,
                    "percentage": 100 * count / total,
                }
            )
    spatial_rows = [
        {
            "tissue_tag": tag,
            "bin_x": x,
            "bin_y": y,
            "bin_z": z,
            "bin_size_mm": options["spatial_bin_mm"],
            "warning_elements": n,
        }
        for (tag, x, y, z), n in sorted(spatial.items())
    ]
    cortical = {}
    for name in interfaces:
        d = np.array(
            [
                r[name + "_distance_mm"] if r[name + "_distance_mm"] is not None else np.nan
                for r in rows
            ]
        )
        cortical[name] = {
            "finite_centroid_distances": int(np.isfinite(d).sum()),
            "within_2mm_centroids": int((d <= 2).sum()),
            "interface_faces": len(interfaces[name].centers),
        }
    report = {
        "evidence_category": "computational_consistency_verified",
        "independent_anatomical_assessment": "pending",
        "tetrahedra_streamed": seen,
        "v1_global_counts_matched": True,
        "global_distributions": "Reused unchanged from v1; not regenerated",
        "reason_for_pass": "v1 did not persist per-element qualities; new tissue counts and locations require one streamed classification pass",
        "warning_elements": len(rows),
        "interfaces": cortical,
        "interface_definition": "CHARM voxel-cell label interfaces; centroid distances, not mesh-node clearance",
        "spatial_definition": "RAS 10mm bins by tissue; bins are not connected components",
        "thresholds_provisional": thresholds,
        "FEM_field_error": "not_estimated",
    }
    return report, per_tissue, spatial_rows, rows, geometry


def GmshChunks(path):
    from causalneurotwin.validation.anatomy.mesh import Gmsh22

    with Gmsh22(path) as mesh:
        yield from mesh.tetra_chunks()


def region_counts(rows, regions, project_root):
    from ..anatomy.io import inside as within
    from ..artifacts import safe as unredirected
    from ..artifacts import sha

    results = []
    for region in regions:
        if (
            region["coordinate_system"] != "native_scanner_RAS_mm"
            or region["mapping_status"] != "computational_consistency_verified"
        ):
            raise ValueError("ROI mapping not verified in native scanner RAS")
        evidence = unredirected(region["mapping_evidence"])
        if not within(evidence, project_root) or sha(evidence) != region["mapping_evidence_sha256"]:
            raise ValueError("ROI mapping evidence missing/changed")
        center = np.asarray(region["center_ras_mm"], float)
        radius = float(region["radius_mm"])
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("Invalid ROI geometry")
        points = np.array([[r[k] for k in ("x_mm", "y_mm", "z_mm")] for r in rows]).reshape(-1, 3)
        selected = np.linalg.norm(points - center, axis=1) <= radius
        results.append(
            {
                "name": region["name"],
                "warning_centroids": int(selected.sum()),
                "radius_mm": radius,
                "mapping_evidence_sha256": sha(evidence),
                "assessment_status": "pending",
                "FEM_field_error": "not_estimated",
            }
        )
    return {
        "status": "measured" if results else "pending",
        "reason": "No verified simulation-region coordinates supplied"
        if not results
        else "Coordinates accepted only against explicit hash-pinned mapping evidence",
        "regions": results,
    }
