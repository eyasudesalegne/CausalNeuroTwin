import numpy as np
import pytest

from causalneurotwin.validation.anatomy.geometry import apply
from causalneurotwin.validation.anatomy_addendum.distance import (
    VoxelBoundary,
    analyze_surface,
    clusters,
    surface_geometry,
)

from .conftest import volume


@pytest.mark.parametrize("spacing", [(1, 1, 1), (2, 3, 4), (0.5, 2, 1)])
@pytest.mark.parametrize("angle", [0, 0.4])
@pytest.mark.parametrize("reflect", [1, -1])
def test_physical_voxel_face_distance(spacing, angle, reflect):
    labels = np.zeros((3, 3, 3))
    labels[1, 1, 1] = 2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    a = np.eye(4)
    a[:3, :3] = rotation @ np.diag(np.array(spacing) * [reflect, 1, 1])
    a[:3, 3] = [8, -4, 13]
    boundary = VoxelBoundary(labels, a, region=2)
    points = apply(a, np.array([[1.7, 1, 1], [1.7, 1.8, 1], [1, 1, 1]]))
    expected = [0.2 * spacing[0], np.hypot(0.2 * spacing[0], 0.3 * spacing[1]), min(spacing) / 2]
    assert np.allclose(boundary.query(points), expected, atol=1e-12)


def test_kdtree_pruning_matches_all_rectangles():
    rng = np.random.default_rng(13)
    labels = rng.integers(0, 3, (4, 5, 6))
    a = np.diag([0.7, 1.3, 2.1, 1])
    b = VoxelBoundary(labels, a, region=2)
    points = rng.uniform(-2, 12, (40, 3))
    q = apply(np.linalg.inv(a), points) * b.spacing
    expected = []
    for p in q:
        half = np.tile(b.spacing / 2, (len(b.centers), 1))
        half[np.arange(len(half)), b.axes] = 0
        expected.append(np.linalg.norm(np.maximum(abs(p - b.centers) - half, 0), axis=1).min())
    assert np.allclose(b.query(points, chunk=7), expected, atol=1e-12)


def test_reject_shear():
    a = np.eye(4)
    a[0, 1] = 0.1
    with pytest.raises(ValueError, match="Sheared"):
        VoxelBoundary(np.zeros((2, 2, 2)), a, region=2)


def test_pair_interface_and_absent_interface():
    labels = np.ones((3, 3, 3))
    labels[1:] = 2
    b = VoxelBoundary(labels, np.eye(4), pairs=[(1, 2)])
    assert b.query([[0, 1, 1]])[0] == 0.5
    assert np.isnan(VoxelBoundary(labels, np.eye(4), pairs=[(2, 3)]).query([[0, 0, 0]])[0])


def test_outside_gm_boundary_includes_image_edge():
    b = VoxelBoundary(np.full((2, 2, 2), 2), np.eye(4), region=2)
    assert np.allclose(b.query([[0, 0, 0], [-1, 0, 0]]), [0.5, 0.5])


def test_offset_sensitivity_and_area_conservation():
    labels = np.ones((5, 5, 7))
    labels[:, :, 3:5] = 2
    v = volume(labels)
    vertices = np.array([[1, 1, 2.4], [2, 1, 2.4], [1, 2, 2.4]])
    faces = np.array([[0, 1, 2]])
    b = VoxelBoundary(labels, np.eye(4), region=2)
    report, a = analyze_surface(v, vertices, faces, b, [-0.5, -0.25, 0, 0.25, 0.5])
    assert report["disagreement_vertices"] == 3
    assert report["persistent_disagreement_vertices"] == 0
    assert report["offset_sensitive_disagreement_vertices"] == 3
    assert np.isclose(report["surface_area_mm2"], 0.5)
    assert np.allclose(a["gm_region_distance_mm"], 0.1)
    _, flipped = analyze_surface(v, vertices, faces[:, ::-1], b, [-0.5, -0.25, 0, 0.25, 0.5])
    assert np.array_equal(a["persistent"], flipped["persistent"])


def test_inside_region_distance_is_zero_but_boundary_distance_is_not():
    labels = np.ones((5, 5, 7))
    labels[:, :, 3:5] = 2
    v = volume(labels)
    vertices = np.array([[1, 1, 3.5], [2, 1, 3.5], [1, 2, 3.5]])
    report, a = analyze_surface(
        v,
        vertices,
        np.array([[0, 1, 2]]),
        VoxelBoundary(labels, np.eye(4), region=2),
        [-0.5, 0, 0.5],
    )
    assert report["disagreement_vertices"] == 0
    assert np.all(a["gm_region_distance_mm"] == 0)
    assert np.allclose(a["signed_boundary_distance_mm"], -1)


def test_undefined_normals_are_indeterminate():
    labels = np.ones((3, 3, 3))
    labels[2] = 2
    v = volume(labels)
    vertices = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], float)
    report, a = analyze_surface(
        v,
        vertices,
        np.array([[0, 1, 2]]),
        VoxelBoundary(labels, np.eye(4), region=2),
        [-0.5, 0, 0.5],
    )
    assert report["offset_indeterminate_vertices"] == 3
    assert not a["persistent"].any()


def test_mesh_connected_clusters_do_not_join_overlapping_banks():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    weights, _, _ = surface_geometry(vertices, faces)
    ids, rows = clusters(vertices, faces, np.ones(6, bool), weights, np.ones(6))
    assert len(rows) == 2
    assert np.isclose(sum(r["area_mm2"] for r in rows), 1)
    assert ids[0] != ids[3]


@pytest.mark.parametrize("offsets", [[0, 0.5], [], [-0.5, 0.5], [0, float("nan")]])
def test_invalid_offsets_rejected(offsets):
    labels = np.ones((3, 3, 3))
    labels[2] = 2
    with pytest.raises(ValueError):
        analyze_surface(
            volume(labels),
            np.eye(3),
            np.array([[0, 1, 2]]),
            VoxelBoundary(labels, np.eye(4), region=2),
            offsets,
        )


def test_outside_fov_not_silently_treated_as_persistent_tissue_defect():
    labels = np.full((4, 4, 4), 2)
    vertices = np.array([[-1, 1, 1], [1, 1, 1], [1, 2, 1]], float)
    report, a = analyze_surface(
        volume(labels),
        vertices,
        np.array([[0, 1, 2]]),
        VoxelBoundary(labels, np.eye(4), region=2),
        [-0.5, 0, 0.5],
    )
    assert report["categories"]["outside_or_nonfinite"]["vertices"] == 1
    assert a["offset_indeterminate"][0] and not a["persistent"][0]
    assert np.isfinite(a["gm_region_distance_mm"][0])
