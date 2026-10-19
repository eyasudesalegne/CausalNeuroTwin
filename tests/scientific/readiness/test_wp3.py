import numpy as np
import pytest

from causalneurotwin.validation.readiness.wp3 import (
    compare_runs,
    condition_plan,
    null_claims,
    numerical_assessment,
    overflow_evidence,
    physical_history,
)


def signal():
    t = np.arange(0.5, 2000, 1.0)
    return t, np.c_[np.sin(2 * np.pi * 10 * t / 1000), 2 * np.sin(2 * np.pi * 12 * t / 1000)]


def test_history_is_identical_at_common_physical_times():
    coarse = np.arange(-10, 0.001, 0.1)
    fine = np.arange(-10, 0.001, 0.05)
    a = physical_history(coarse, 3, 20)
    b = physical_history(fine, 3, 20)
    assert np.allclose(a, b[::2], atol=1e-12)
    assert not np.array_equal(a, physical_history(coarse, 3, 21))


def test_convergence_acceptance_and_phase_sensitivity(options):
    t, x = signal()
    crit = options["wp3"]["thresholds"]
    assert compare_runs((t, x), (t, x * 1.01), crit)["status"] == "numerical_pass"
    shifted = np.roll(x, 20, axis=0)
    assert compare_runs((t, x), (t, shifted), crit, True)["status"] == "numerical_fail"
    assert compare_runs((t, x), (t, shifted), crit, False)["status"] == "numerical_pass"


@pytest.mark.parametrize("bad", ["time", "nan", "layout", "zero"])
def test_bad_comparisons_are_not_accepted(options, bad):
    t, x = signal()
    tt = t.copy()
    xx = x.copy()
    if bad == "time":
        tt += 0.5
    if bad == "nan":
        xx[0, 0] = np.nan
    if bad == "layout":
        xx = xx[:, :1]
    if bad == "zero":
        assert (
            compare_runs((t, x * 0), (t, x * 0), options["wp3"]["thresholds"])["status"]
            == "indeterminate"
        )
        return
    with pytest.raises(ValueError):
        compare_runs((t, x), (tt, xx), options["wp3"]["thresholds"])


def test_bounded_numerical_design_is_not_a_coupling_speed_grid(options):
    plan = condition_plan(options["wp3"])
    assert len(plan) == 8 and len({r["name"] for r in plan}) == 8
    assert {r["dt"] for r in plan} == {0.1, 0.05, 0.025}
    assert {r["warm"] for r in plan} == {1000, 2000, 4000}


def test_warmup_overflow_does_not_become_clean_pass():
    r = numerical_assessment({"test": {"status": "numerical_pass"}}, [{"runtime_warning_count": 3}])
    assert r["status"] == "indeterminate"
    assert (
        numerical_assessment({"test": {"status": "numerical_fail"}}, [])["status"]
        == "numerical_fail"
    )


@pytest.mark.parametrize("contradiction", [False, True])
def test_null2_distribution_claims_match_construction(contradiction):
    single = {
        "node_weighted_strength_max_relative_error": 1e-12,
        "node_weighted_strength_tolerance": 1e-10,
        "global_original_weight_distribution_preserved": contradiction,
        "weight_length_bundle_distribution_preserved": False,
        "degree_sequence_preserved_exactly": True,
    }
    ens = {
        "maximum_strength_relative_error_across_ensemble": 1e-12,
        "degree_sequence_preserved": True,
        "tract_length_distribution_preserved_exactly_per_member": True,
    }
    member = {
        "weight_distribution_preserved_after_strength_balance": False,
        "weight_length_bundle_distribution_preserved_after_balance": False,
    }
    r = null_claims(single, ens, member)
    assert r["provenance_consistent"] is not contradiction
    assert r["claims"]["pure_topology_intervention"] is False
    assert r["claims"]["endpoint_specific_delays"] == "not_preserved"


def test_saved_overflow_audit_is_reused_not_repeated():
    r = overflow_evidence(
        "RuntimeWarning: overflow encountered in exp",
        "Warm-up warnings : 5\nContinuation warnings : 0",
    )
    assert r["warmup_warnings"] == 5 and r["continuation_warnings"] == 0
    assert "not evidence" in r["G40_warmup_policy"]


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
def test_history_grid_aligns_monitor_clock_without_time_interpolation(dt):
    from causalneurotwin.validation.readiness.wp3 import history_grid

    t = history_grid(52.82275, dt, 1.0)
    assert np.isclose(t[0], -54) and t[-1] == 0
    assert np.isclose((len(t) - 1) * dt, 54)
    assert len(t) * dt > 52.82275


def test_noninteger_monitor_grid_refused():
    from causalneurotwin.validation.readiness.wp3 import history_grid

    with pytest.raises(ValueError):
        history_grid(10, 0.3, 1)


def test_missing_numerical_evidence_cannot_pass():
    assert numerical_assessment({}, [])["status"] == "indeterminate"
    assert (
        numerical_assessment({"x": {"status": "numerical_pass"}}, [])["status"] == "indeterminate"
    )
