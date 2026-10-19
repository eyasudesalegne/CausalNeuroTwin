import numpy as np
import pytest

from causalneurotwin.validation.anatomy.mesh import tetra_quality
from causalneurotwin.validation.anatomy_addendum.distance import VoxelBoundary
from causalneurotwin.validation.anatomy_addendum.mesh import (
    flags,
    region_counts,
    stream_localization,
)
from causalneurotwin.validation.anatomy_addendum.parcels import parcel_masks

from .conftest import REGULAR, volume


@pytest.mark.parametrize("aid", range(1, 33))
def test_all_32_parcel_layouts_reuse_distance_and_expose_rejected_sample(aid):
    nleft = 32
    direct = np.zeros(64, int)
    final = np.zeros(64, int)
    distance = np.zeros(64)
    correct = 0 if aid % 2 else 32
    wrong = 32 if aid % 2 else 0
    direct[correct] = aid
    direct[wrong] = aid
    final[correct : correct + 2] = aid
    distance[correct + 1] = 12
    row = {
        "atlas_id": aid,
        "label": "test" + ("_L" if aid % 2 else "_R"),
        "seed_vertices": 1,
        "final_vertices": 2,
        "wrong_hemisphere_samples": 1,
        "added_gt10mm": 1,
    }
    seeds, rejected, added, r = parcel_masks(row, direct, final, distance, nleft)
    assert seeds[correct] and rejected[wrong] and added[correct + 1]
    assert r["added_distance_max_mm"] == 12
    assert r["assessment_status"] == "pending"


@pytest.mark.parametrize(
    "change",
    ["seed_distance", "wrong_count", "negative_distance", "final_wrong_side", "missing_seed"],
)
def test_inconsistent_stored_parcel_evidence_rejected(change):
    row = {
        "atlas_id": 1,
        "label": "X_L",
        "seed_vertices": 1,
        "final_vertices": 2,
        "wrong_hemisphere_samples": 0,
        "added_gt10mm": 1,
    }
    direct = np.array([1, 0, 0, 0])
    final = np.array([1, 1, 0, 0])
    d = np.array([0, 12, 0, 0], float)
    if change == "seed_distance":
        d[0] = 1
    if change == "wrong_count":
        row["wrong_hemisphere_samples"] = 2
    if change == "negative_distance":
        d[1] = -2
    if change == "final_wrong_side":
        final[2] = 1
    if change == "missing_seed":
        direct[0] = 0
    with pytest.raises(ValueError):
        parcel_masks(row, direct, final, d, 2)


def test_new_mesh_tissue_counts_missing_interface_is_null(tmp_path):
    vertices = np.concatenate([REGULAR, REGULAR + [3, 0, 0]])
    vertices[7, 2] = 0.000001
    path = tmp_path / "test.msh"
    with path.open("w") as h:
        h.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n8\n")
        for i, xyz in enumerate(vertices, 1):
            h.write(str(i) + " " + " ".join(map(str, xyz)) + "\n")
        h.write("$EndNodes\n$Elements\n2\n10 4 2 1 1 1 2 3 4\n20 4 2 7 7 5 6 7 8\n$EndElements\n")
    q = tetra_quality(vertices.reshape(2, 4, 3))
    thresholds = {
        "mean_ratio_below": 0.1,
        "scaled_jacobian_below": 0.2,
        "min_dihedral_below_deg": 5,
        "max_dihedral_above_deg": 175,
    }
    prior = {
        "tetrahedra_evaluated": 2,
        "triage_thresholds_provisional": thresholds,
        "triage_counts": {k: int(v.sum()) for k, v in flags(q, thresholds).items()},
    }
    labels = np.full((8, 8, 8), 2)
    v = volume(labels)
    interfaces = {"GM_CSF": VoxelBoundary(labels, np.eye(4), pairs=[(2, 3)])}
    report, counts, _bins, rows, geometry = stream_localization(
        path, prior, interfaces, v, {"spatial_bin_mm": 10}, {10, 20}
    )
    assert report["warning_elements"] == 1 and rows[0]["tissue_tag"] == 7
    assert rows[0]["GM_CSF_distance_mm"] is None
    assert (
        next(
            r for r in counts if r["tissue_tag"] == 7 and r["criterion"] == "scaled_jacobian_below"
        )["percentage"]
        == 100
    )
    assert (
        next(r for r in counts if r["tissue_tag"] == 1 and r["criterion"] == "any_warning")["count"]
        == 0
    )
    assert set(geometry) == {10, 20}
    prior["triage_counts"]["scaled_jacobian_below"] = 0
    with pytest.raises(ValueError, match="differs"):
        stream_localization(path, prior, interfaces, v, {"spatial_bin_mm": 10}, {10, 20})


def test_simulation_roi_absence_pending():
    assert region_counts([], [], "/unused")["status"] == "pending"


def test_unverified_simulation_roi_rejected():
    with pytest.raises(ValueError, match="mapping"):
        region_counts(
            [], [{"coordinate_system": "unknown", "mapping_status": "pending"}], "/unused"
        )
