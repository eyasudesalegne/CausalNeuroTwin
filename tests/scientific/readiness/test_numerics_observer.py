from types import SimpleNamespace as NS

import numpy as np
import pytest

from causalneurotwin.validation.readiness.numerics import EvaluationAudit


def fixture(tmp_path, bad=False):
    m = NS(
        r=np.array([0.56]),
        v0=np.array([5.52]),
        a_1=np.array([1.0]),
        a_3=np.array([0.25]),
        J=np.array([135.0]),
    )
    m.dfun = lambda state, *args: np.full_like(state, np.nan) if bad else state * 2
    c = NS(r=np.array([0.56]), midpoint=np.array([6.0]), pre=lambda a, b: b[:, :1] * 3)
    return m, c, EvaluationAudit(m, c, tmp_path)


def test_observer_preserves_values_and_records_predictor_calls(tmp_path):
    m, c, a = fixture(tmp_path)
    y = np.ones((6, 3, 1))
    assert np.array_equal(m.dfun(y, None), 2 * y)
    assert np.array_equal(m.dfun(y + 2, None), 2 * (y + 2))
    x = np.ones((3, 2, 3, 1))
    assert np.array_equal(c.pre(x, x), x[:, :1] * 3)
    assert a.snapshot()["rhs_calls"] == 2 and a.snapshot()["coupling_pre_calls"] == 1


def test_overflow_exponent_recorded_without_changing_model(tmp_path):
    m, _c, a = fixture(tmp_path)
    y = np.zeros((6, 3, 1))
    y[1] = -2000
    expected = y * 2
    assert np.array_equal(m.dfun(y, None), expected)
    assert a.snapshot()["overflow_exponents"] > 0


@pytest.mark.parametrize("fault", ["input", "derivative", "coupling"])
def test_nonfinite_evaluations_fail_and_preserve_diagnostics(tmp_path, fault):
    m, c, _a = fixture(tmp_path, fault == "derivative")
    y = np.ones((6, 3, 1))
    if fault == "input":
        y[0] = np.inf
    with pytest.raises(FloatingPointError):
        if fault == "coupling":
            c.pre(None, np.full((3, 2, 3, 1), np.nan))
        else:
            m.dfun(y, None)
    assert (tmp_path / "numerical_failure.json").is_file()
