import numpy as np
import pytest

from causalneurotwin.validation.readiness.wp2 import (
    compare_connectivity,
    edge_uncertainty,
    fsl_to_world,
    gradients,
    motion_plan,
)


@pytest.mark.parametrize("reflect", [1, -1])
@pytest.mark.parametrize("angle", [0, 0.4])
def test_fsl_world_coordinate_convention(reflect, angle):
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    a = np.eye(4)
    a[:3, :3] = r @ np.diag([reflect * 2, 3, 4])
    result = fsl_to_world(np.eye(3), a)
    # FSL handedness correction makes world x consistent across these two storage choices.
    assert np.allclose(result[0], -r[:, 0])
    assert np.allclose(result[1:], r[:, 1:].T)


def sample_gradients(opt):
    b = np.array([0, 1000, 1000, 1000.0])
    v = np.c_[np.zeros(3), np.eye(3)]
    a = np.eye(4)
    m = np.c_[fsl_to_world(v.T, a), b]
    return b, v, m, a


def test_gradient_row_order_and_antipodal_equivalence(options):
    opt = options["wp2"]
    b, v, m, a = sample_gradients(opt)
    m[2, :3] *= -1
    r, _ = gradients(b, v, m, a, 4, b, ["synthetic.nii.gz"], ["synthetic.nii.gz"] * 4, opt)
    assert r["FSL_MRtrix_consistent"] and r["b0_count"] == 1
    with pytest.raises(ValueError, match="aligned"):
        gradients(b, v, m, a, 4, b[::-1], ["synthetic.nii.gz"], ["synthetic.nii.gz"] * 4, opt)


@pytest.mark.parametrize("change", ["count", "norm", "nan", "source", "btable"])
def test_bad_gradient_provenance_rejected(options, change):
    opt = options["wp2"]
    b, v, m, a = sample_gradients(opt)
    n = 4
    sources = ["synthetic"]
    if change == "count":
        n = 3
    if change == "norm":
        v[0, 1] = 2
    if change == "nan":
        v[1, 2] = np.nan
    if change == "source":
        sources = ["other"]
    if change == "btable":
        m[1, 3] = 2000
    with pytest.raises(ValueError):
        gradients(b, v, m, a, n, b, sources, ["synthetic"] * 4, opt)


def test_axis_mismatch_is_reported_without_automatic_repair(options):
    opt = options["wp2"]
    b, v, m, a = sample_gradients(opt)
    m[1, :3] = [0, 1, 0]
    report, _ = gradients(b, v, m, a, 4, b, ["S"], ["S"] * 4, opt)
    assert not report["FSL_MRtrix_consistent"]


def sampling(opt):
    rng = np.random.default_rng(4)
    directions = rng.normal(size=(24, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    g = np.r_[np.repeat(directions, 2, axis=0), np.zeros((4, 3))]
    b = np.r_[np.full(48, 1000), np.zeros(4)]
    fd = np.r_[np.tile([0.1, 2.0], 24), np.full(4, 0.1)]
    opt = opt.copy()
    opt.update(minimum_directions=12, minimum_retained_fraction=0.4)
    return b, g, fd, opt


def test_three_conditions_only_when_sampling_and_provenance_gates_pass(options):
    b, g, fd, opt = sampling(options["wp2"])
    r = motion_plan(b, g, fd, opt, True, True)
    assert r["status"] == "feasible_requires_approval"
    assert len(r["conditions"]) == 3
    assert len(r["conditions"][1]["indices"]) == len(r["conditions"][2]["indices"])
    assert len(set(r["conditions"][2]["indices"])) == len(r["conditions"][2]["indices"])
    blocked = motion_plan(b, g, fd, opt, False, False)
    assert blocked["status"] == "blocked" and len(blocked["conditions"]) == 1


@pytest.mark.parametrize("case", ["motion", "coverage", "b0", "contrast"])
def test_infeasible_subsets_do_not_relax_thresholds(options, case):
    b, g, fd, opt = sampling(options["wp2"])
    if case == "motion":
        fd[:] = 3
    if case == "coverage":
        g[b > 0] = [1, 0, 0]
    if case == "b0":
        fd[b == 0] = 3
    if case == "contrast":
        fd[:] = 0.1
    r = motion_plan(b, g, fd, opt, True, True)
    assert r["status"] == "blocked" and len(r["conditions"]) == 1
    assert r["no_automatic_threshold_relaxation"]


def test_endpoint_pass_difference_is_not_false_positive_rate(connectomes):
    a, b, la, lb = connectomes
    report, _rows = compare_connectivity(a, b, la, lb, [1, 2, 3], [1, 2, 3])
    assert report["jaccard"] == 2 / 3 and report["pass_only"] == 1
    assert report["false_positive_or_negative_rate"] == "not_identifiable"
    with pytest.raises(ValueError):
        compare_connectivity(a, b, la, lb, [2, 1, 3], [1, 2, 3])


def test_missing_edges_have_zero_weight_but_no_invented_length(connectomes):
    a, b, la, lb = connectomes
    r = edge_uncertainty([a, b], [la, lb], [[1, 2, 3]] * 2, 10)
    edge = next(x for x in r["edge_rows"] if x["i"] == 0 and x["j"] == 2)
    assert edge["presence_fraction"] == 0.5 and edge["weight_mean"] == 0.2
    assert edge["length_observations"] == 1 and edge["length_min_mm"] == 30
    assert "not_estimated" in r["confidence_intervals"]
    with pytest.raises(ValueError):
        edge_uncertainty([a] * 4, [la] * 4, [[1, 2, 3]] * 4, 10)


def test_shear_not_silently_approximated():
    a = np.eye(4)
    a[0, 1] = 0.2
    with pytest.raises(ValueError):
        fsl_to_world(np.eye(3), a)


@pytest.mark.parametrize("scale", [np.nan, np.inf])
def test_nonfinite_reference_scaling_rejected(connectomes, scale):
    a, b, la, lb = connectomes
    with pytest.raises(ValueError):
        edge_uncertainty([a, b], [la, lb], [[1, 2, 3]] * 2, scale)
