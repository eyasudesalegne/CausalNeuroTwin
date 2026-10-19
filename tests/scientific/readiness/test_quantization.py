import numpy as np
import pytest

from causalneurotwin.validation.readiness.history_init import (
    coupling_comparison,
    exact_storage_comparison,
)


def test_exact_float32_representation_not_float64_equality():
    x = np.array([0.123456789, -0.00987654321, np.finfo(np.float32).tiny / 3])
    stored = x.astype(np.float32)
    assert not np.array_equal(x, stored)
    r = exact_storage_comparison(stored, x)
    assert r["passed"] and r["reference_quantization_max_abs"] > 0


def test_any_storage_ulp_change_fails():
    x = np.array([0.123456789])
    stored = x.astype(np.float32)
    stored[0] = np.nextafter(stored[0], np.float32(np.inf))
    r = exact_storage_comparison(stored, x)
    assert not r["passed"] and r["mismatched_elements"] == 1


@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "inf", "permutation", "sign"])
def test_invalid_storage_cannot_hide_behind_tolerance(fault):
    x = np.array([0.123456789, -0.003142])
    a = x.astype(np.float32)
    if fault == "dtype":
        a = a.astype(float)
    if fault == "shape":
        a = a[:, None]
    if fault == "nan":
        a[0] = np.nan
    if fault == "inf":
        a[0] = np.inf
    if fault == "permutation":
        a = a[::-1]
    if fault == "sign":
        a = -a
    assert not exact_storage_comparison(a, x)["passed"]


def test_same_rounding_bin_is_not_extra_ulp_slack():
    x = np.float32(0.0123456)
    spacing = float(np.spacing(x))
    assert exact_storage_comparison(np.array([x]), [float(x) + 0.1 * spacing])["passed"]
    assert not exact_storage_comparison(np.array([x]), [float(x) + 0.75 * spacing])["passed"]


def test_float64_coupling_bound_inside_outside_and_no_broadcast():
    e = np.ones((2, 3, 1))
    budget = coupling_comparison(e, e, 3)["max_abs_error_limit"]
    assert coupling_comparison(e + budget / 2, e, 3)["passed"]
    assert not coupling_comparison(e + budget * 2, e, 3)["passed"]
    assert not coupling_comparison(e[:1], e, 3)["passed"]
