import numpy as np
import pytest

from causalneurotwin.validation.structural.atlas import support
from causalneurotwin.validation.structural.observation import (
    aggregate,
    operator_difference,
    registration_gate,
    unresolved_bracket,
)
from causalneurotwin.validation.structural.prediction import (
    hierarchical_uncertainty,
    paired_improvement,
    response_metrics,
    validate_splits,
)


def gain_data(index, data):
    G = np.vstack([np.arange(1, 9), -np.arange(1, 9)]).astype(float)
    _, den = support(index, data["direct"], data["original_trusted"], data["area"])
    return G, den


def test_observation_keeps_noncortical_nodes_zero(index, data):
    G, den = gain_data(index, data)
    out, rep = aggregate(index, G, data["direct"], data["original_trusted"], data["area"], den)
    assert out.shape == (2, 5) and (out[:, -1] == 0).all()
    assert rep["noncortical_nodes_remain_in_dynamics"]
    assert not rep["JR_to_dipole_amplitude_calibrated"]
    assert rep["column_mass"] == [1, 1, 1, 1, 0]


def test_omission_vs_renormalization_are_explicit(index, data):
    G, den = gain_data(index, data)
    mask = data["original_trusted"].copy()
    mask[0] = False
    a, ra = aggregate(index, G, data["direct"], mask, data["area"], den)
    b, rb = aggregate(index, G, data["direct"], mask, data["area"], den, "retained_area")
    assert ra["column_mass"][0] == 2 / 3 and rb["column_mass"][0] == 1
    assert not np.array_equal(a, b)


def test_uncertain_contribution_bracket_linear(index, data):
    G, den = gain_data(index, data)
    unresolved = np.zeros(8, bool)
    unresolved[0] = True
    core = data["original_trusted"] & ~unresolved
    a = unresolved_bracket(index, G, data["direct"], core, unresolved, data["area"], den, 0)
    b = unresolved_bracket(index, G, data["direct"], core, unresolved, data["area"], den, 1)
    midpoint = unresolved_bracket(
        index, G, data["direct"], core, unresolved, data["area"], den, 0.5
    )
    full, _ = aggregate(index, G, data["direct"], data["original_trusted"], data["area"], den)
    assert np.allclose(b, full) and np.allclose(midpoint, (a + b) / 2)
    with pytest.raises(ValueError):
        unresolved_bracket(index, G, data["direct"], core, core, data["area"], den, 0.5)


@pytest.mark.parametrize("bad", ["gain_nan", "denom", "policy", "shape"])
def test_operator_guards(index, data, bad):
    G, den = gain_data(index, data)
    policy = "fixed_baseline_area"
    if bad == "gain_nan":
        G[0, 0] = np.nan
    elif bad == "denom":
        den[0] = 0
    elif bad == "shape":
        G = G[:, :-1]
    else:
        policy = "silent_retune"
    with pytest.raises(ValueError):
        aggregate(index, G, data["direct"], data["original_trusted"], data["area"], den, policy)


def test_operator_difference_undefined_zero_columns():
    r = operator_difference(np.array([[1.0, 0], [2.0, 0]]), np.array([[2.0, 0], [4.0, 0]]))
    assert r["relative_frobenius"] == 1
    assert np.isnan(r["column_relative_difference"][1])


def registration():
    return dict(
        decision="ACCEPT",
        independent_review=True,
        leadfield_sha256="a" * 64,
        reviewer="Synthetic reviewer",
        evidence="synthetic.txt",
        reference="average",
        channel_order_verified=True,
        source_order_verified=True,
    )


def test_registration_exact_forward_gate():
    registration_gate(registration(), "a" * 64)
    with pytest.raises(RuntimeError):
        registration_gate(registration(), "b" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", ""),
        ("independent_review", False),
        ("reviewer", ""),
        ("reference", "unknown"),
        ("channel_order_verified", False),
        ("source_order_verified", False),
    ],
)
def test_registration_incomplete(field, value):
    e = registration()
    e[field] = value
    with pytest.raises(RuntimeError):
        registration_gate(e, "a" * 64)


def splits():
    return [
        dict(session="s1", run="r1", condition="a", split="calibration"),
        dict(session="s1", run="r2", condition="b", split="validation"),
        dict(session="s2", run="r1", condition="c", split="test"),
    ]


def test_split_registry_no_invented_condition():
    assert validate_splits(splits(), True)["runs"] == 3


@pytest.mark.parametrize(
    "kind", ["run_leakage", "condition_leakage", "empty_identity", "no_test", "none"]
)
def test_split_leakage_failures(kind):
    rows = splits()
    if kind == "run_leakage":
        rows[1]["run"] = "r1"
    elif kind == "condition_leakage":
        rows[2]["condition"] = "a"
    elif kind == "empty_identity":
        rows[0]["session"] = ""
    elif kind == "no_test":
        rows.pop()
    else:
        rows = []
    with pytest.raises(ValueError):
        validate_splits(rows, True)


def waveform():
    x = np.array([1.0, 2, 4, 3, 1, 0])
    return np.column_stack([x, -x]), np.arange(6.0), np.ones(6, bool), ["C1", "C2"]


def test_identical_waveform_without_calibration_cannot_claim_amplitude_accuracy():
    y, t, m, c = waveform()
    r = response_metrics(y, y, t, m, c, c)
    assert r["waveform_rmse"] is None
    assert r["temporal_correlation_median"] == 1
    assert r["topography_correlation_median"] == 1
    assert r["gfp_peak_latency_error_ms"] == 0 and not r["fit_performed"]


def test_calibrated_waveform_metrics():
    y, t, m, c = waveform()
    r = response_metrics(2 * y, y, t, m, c, c, units_verified=True)
    assert np.isclose(r["waveform_nrmse"], 1)
    assert r["waveform_rmse"] > 0


def test_artifact_mask_does_not_fit_or_interpolate():
    y, t, m, c = waveform()
    p = y.copy()
    p[0] = np.nan
    m[0] = False
    assert response_metrics(p, y, t, m, c, c, True)["waveform_rmse"] == 0


@pytest.mark.parametrize("kind", ["channels", "time", "reference", "nonfinite", "window"])
def test_waveform_alignment_guard(kind):
    y, t, m, c = waveform()
    p = y.copy()
    pc = c.copy()
    if kind == "channels":
        pc = pc[::-1]
    elif kind == "time":
        t[3] = t[2]
    elif kind == "reference":
        p += 1
    elif kind == "nonfinite":
        p[2, 0] = np.nan
    else:
        m[:4] = False
    with pytest.raises(ValueError):
        response_metrics(p, y, t, m, pc, c)


def test_paired_improvement_not_unpaired_selection():
    assert np.array_equal(paired_improvement([3, 4], [2, 2], ["r1", "r2"], ["r1", "r2"]), [1, 2])
    with pytest.raises(ValueError):
        paired_improvement([3, 4], [2, 2], ["r1", "r2"], ["r2", "r1"])


def test_single_session_uncertainty_indeterminate():
    r = hierarchical_uncertainty([1, 2], ["s1", "s1"], ["r1", "r2"], draws=50)
    assert r["status"] == "indeterminate"


def test_hierarchical_uncertainty_session_balanced_not_trial_pseudoreplication():
    r = hierarchical_uncertainty(
        [1, 1, 1, 3], ["s1", "s1", "s1", "s2"], ["r1", "r1", "r1", "r1"], draws=100
    )
    assert r["mean_session_balanced_improvement"] == 2
    assert r["sessions"] == 2
    assert r["status"] == "descriptive_conditional_interval"
