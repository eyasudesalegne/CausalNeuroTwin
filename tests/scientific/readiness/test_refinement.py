import pytest

from causalneurotwin.validation.readiness.wp3 import numerical_assessment, refinement_assessment


def metric(x):
    return {"metrics": {"regional_nrmse_p95": x}}


@pytest.mark.parametrize(
    "coarse,fine,passed", [(0.04, 0.01, True), (0.01, 0.04, False), (1e-9, 2e-9, True)]
)
def test_predefined_refinement_gate(coarse, fine, passed):
    r = refinement_assessment(
        metric(coarse), metric(fine), {"relative_slack": 0.05, "absolute_floor": 1e-7}
    )
    assert (r["status"] == "numerical_pass") == passed


def test_missing_refinement_is_indeterminate():
    assert refinement_assessment({}, metric(0.1), {})["status"] == "indeterminate"


def test_initial_condition_sensitivity_is_not_solver_accuracy_failure():
    r = numerical_assessment(
        {
            "dt_a": {"status": "numerical_pass"},
            "initial_seed_B_statistics": {"status": "numerical_fail"},
        },
        [{"runtime_warning_count": 0}],
    )
    assert r["status"] == "robustness_sensitive" and not r["accuracy_failed"]
