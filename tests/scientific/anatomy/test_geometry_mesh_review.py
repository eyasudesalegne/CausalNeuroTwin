import json
import socket

import numpy as np
import pytest
from scipy.io import savemat

from causalneurotwin.validation.anatomy.geometry import (
    apply,
    itk_matrix,
    point_map,
    validate_affine,
)
from causalneurotwin.validation.anatomy.io import (
    CONSISTENCY,
    INDEPENDENT,
    PENDING,
    compute_guard,
    digest,
    write_csv,
    write_json,
)
from causalneurotwin.validation.anatomy.mesh import Gmsh22, tetra_quality
from causalneurotwin.validation.anatomy.review import REVIEW_FIELDS, assess_review

from .conftest import REGULAR, write_gmsh


@pytest.mark.parametrize("matrix", [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)])
def test_bad_affine_rejected(matrix):
    with pytest.raises(ValueError):
        validate_affine(matrix)


@pytest.mark.parametrize("zyx", [0, 1])
def test_itk_euler_center_order_and_lps_to_ras(tmp_path, zyx):
    p = tmp_path / "euler.mat"
    angles = [np.pi / 2, np.pi / 2, 0]
    center = np.array([2.0, 3.0, 4.0])
    t = np.array([1.0, 2.0, 3.0])
    savemat(
        p,
        {
            "Euler3DTransform_double_3_3": np.r_[angles, t][:, None],
            "fixed": np.r_[center, zyx][:, None],
        },
        format="4",
    )
    lps_point = center + np.array([1.0, 0, 0])
    # RxRy rotates this vector to +y; RyRx rotates it to -z.
    expected = center + t + (np.array([0, 0, -1]) if zyx else np.array([0, 1, 0]))
    flip = np.array([-1, -1, 1])
    assert np.allclose(apply(itk_matrix(p), lps_point * flip), expected * flip)


def test_itk_affine_center_and_translation(tmp_path):
    p = tmp_path / "affine.mat"
    savemat(
        p,
        {
            "AffineTransform_double_3_3": np.r_[np.diag([2.0, 3.0, 4.0]).ravel(), [1, 2, 3]][
                :, None
            ],
            "fixed": np.array([[2.0], [3.0], [4.0]]),
        },
        format="4",
    )
    assert np.allclose(apply(itk_matrix(p), [-2, -3, 4]), [-3, -5, 7])


def test_point_pull_direction_is_inverse_for_native_to_acpc():
    m = np.eye(4)
    m[:3, 3] = [10, 20, 30]
    assert np.allclose(apply(point_map("native", "acpc", m), [10, 20, 30]), [0, 0, 0])
    assert np.allclose(apply(point_map("acpc", "native", m), [0, 0, 0]), [10, 20, 30])


def test_unsupported_itk_transform_fails(tmp_path):
    p = tmp_path / "unsupported.mat"
    savemat(
        p, {"BSplineTransform_double_3_3": np.zeros((6, 1)), "fixed": np.zeros((4, 1))}, format="4"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        itk_matrix(p)


def test_regular_tetra_has_normalized_shape():
    q = tetra_quality(REGULAR[None])
    assert q["mean_ratio"][0] == pytest.approx(1)
    assert q["scaled_jacobian"][0] == pytest.approx(1)
    assert q["edge_ratio"][0] == pytest.approx(1)
    assert q["min_dihedral_deg"][0] == pytest.approx(np.degrees(np.arccos(1 / 3)))
    assert q["max_dihedral_deg"][0] == pytest.approx(q["min_dihedral_deg"][0])


def test_shape_invariant_under_scale_and_translation():
    a = tetra_quality(REGULAR[None])
    b = tetra_quality((REGULAR * 7 + [100, -20, 9])[None])
    for key in a:
        assert np.allclose(a[key], b[key])


def test_sliver_and_inversion_detected():
    thin = REGULAR.copy()
    thin[3, 2] = 1e-8
    q = tetra_quality(thin[None])
    assert q["mean_ratio"][0] < 1e-4 and q["min_dihedral_deg"][0] < 0.01
    inverted = REGULAR[[1, 0, 2, 3]]
    assert tetra_quality(inverted[None])["scaled_jacobian"][0] == pytest.approx(-1)


def test_degeneracy_serializes_as_null_not_nonstandard_json(tmp_path):
    q = tetra_quality(np.zeros((1, 4, 3)))
    p = tmp_path / "result.json"
    write_json(p, q)
    assert "NaN" not in p.read_text() and "Infinity" not in p.read_text()
    assert json.loads(p.read_text())["mean_ratio"] == [None]


@pytest.mark.parametrize("binary,endian", [(True, "<"), (True, ">"), (False, "<")])
def test_gmsh_formats_noncontiguous_ids_and_streaming(tmp_path, binary, endian):
    p = write_gmsh(
        tmp_path / "mesh.msh", np.concatenate([REGULAR, REGULAR + [2, 0, 0]]), binary, endian
    )
    with Gmsh22(p) as mesh:
        chunks = list(mesh.tetra_chunks(1))
    assert len(chunks) == 2
    assert np.allclose(chunks[0][2][0], REGULAR)
    assert chunks[1][0].tolist() == [101] and chunks[1][1].tolist() == [2]


def test_truncated_mesh_fails(tmp_path):
    p = write_gmsh(tmp_path / "mesh.msh", REGULAR)
    p.write_bytes(p.read_bytes()[:-12])
    with pytest.raises(ValueError), Gmsh22(p) as mesh:
        list(mesh.tetra_chunks())


def test_mesh_unknown_element_type_fails(tmp_path):
    p = tmp_path / "mesh.msh"
    p.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n10 0 0 0\n$EndNodes\n$Elements\n1\n1 99 0 10\n$EndElements\n"
    )
    with pytest.raises(ValueError), Gmsh22(p) as mesh:
        list(mesh.tetra_chunks())


def test_compute_guard_rejects_login_even_with_job_id(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "cn0001")
    monkeypatch.setattr(socket, "gethostname", lambda: "login1.discoverer")
    with pytest.raises(RuntimeError, match="login"):
        compute_guard()


def test_compute_guard_rejects_unallocated_execution(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match=r"[Aa]llocated"):
        compute_guard()


def review_file(path, **override):
    row = {k: "" for k in REVIEW_FIELDS}
    row.update(
        item_id="insula_left",
        domain="surfaces",
        decision="accept",
        reason="Reviewed boundary",
        evidence_image="figures/evidence.png",
        reviewer_id="R1",
        reviewed_at_utc="2026-10-17T12:00:00Z",
        independence_basis="independently_inspected_original_T1",
    )
    row.update(override)
    write_csv(path, [row], REVIEW_FIELDS)
    return path


def test_reviewer_acceptance_requires_image_and_attestation(tmp_path):
    p = review_file(tmp_path / "r.csv")
    report = assess_review(p, [{"item_id": "insula_left", "domain": "surfaces"}], tmp_path, {})
    assert report["items"][0]["assessment_status"] == "indeterminate"


def test_scoped_review_pass_does_not_become_whole_subject_biological_pass(tmp_path):
    (tmp_path / "figures").mkdir()
    im = tmp_path / "figures/evidence.png"
    im.write_bytes(b"synthetic")
    p = review_file(tmp_path / "r.csv")
    report = assess_review(
        p,
        [{"item_id": "insula_left", "domain": "surfaces"}],
        tmp_path,
        {"figures/evidence.png": digest(im)},
    )
    assert report["items"][0]["evidence_category"] == INDEPENDENT
    assert report["whole_subject_biological_validation"] == "not_established"
    assert report["evidence_category"] == PENDING


def test_left_right_header_alone_cannot_pass(tmp_path):
    (tmp_path / "figures").mkdir()
    im = tmp_path / "figures/evidence.png"
    im.write_bytes(b"synthetic")
    p = review_file(tmp_path / "r.csv", item_id="orientation_lr", domain="orientation")
    report = assess_review(
        p,
        [{"item_id": "orientation_lr", "domain": "orientation"}],
        tmp_path,
        {"figures/evidence.png": digest(im)},
    )
    assert report["passed_items"] == 0


def test_changed_image_invalidates_acceptance(tmp_path):
    (tmp_path / "figures").mkdir()
    im = tmp_path / "figures/evidence.png"
    im.write_bytes(b"before")
    before = digest(im)
    im.write_bytes(b"after")
    p = review_file(tmp_path / "r.csv")
    report = assess_review(
        p,
        [{"item_id": "insula_left", "domain": "surfaces"}],
        tmp_path,
        {"figures/evidence.png": before},
    )
    assert report["passed_items"] == 0


def test_mesh_human_acceptance_is_technical_only(tmp_path):
    (tmp_path / "figures").mkdir()
    im = tmp_path / "figures/evidence.png"
    im.write_bytes(b"synthetic")
    p = review_file(
        tmp_path / "r.csv",
        item_id="mesh_shape_candidates",
        domain="mesh",
        independence_basis="technical_mesh_review",
    )
    report = assess_review(
        p,
        [{"item_id": "mesh_shape_candidates", "domain": "mesh"}],
        tmp_path,
        {"figures/evidence.png": digest(im)},
    )
    assert report["passed_items"] == 0 and report["technical_reviewed_items"] == 1
    assert report["items"][0]["evidence_category"] == CONSISTENCY


def test_derivative_agreement_is_not_independent_assessment(tmp_path):
    (tmp_path / "figures").mkdir()
    im = tmp_path / "figures/evidence.png"
    im.write_bytes(b"synthetic")
    p = review_file(tmp_path / "r.csv", independence_basis="two_derivatives_agree")
    report = assess_review(
        p,
        [{"item_id": "insula_left", "domain": "surfaces"}],
        tmp_path,
        {"figures/evidence.png": digest(im)},
    )
    assert report["passed_items"] == 0
