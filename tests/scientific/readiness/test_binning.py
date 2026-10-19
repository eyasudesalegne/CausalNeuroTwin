import numpy as np
import pytest

from causalneurotwin.validation.readiness.binning import PhysicalBins


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
def test_matched_linear_integrals_and_centers(dt):
    bins = PhysicalBins(dt, 1, 2000, np.array([2000.0, 4000.0]))
    output = []
    for step in range(1, round(2 / dt) + 1):
        t = 2000 + step * dt
        r = bins.push(t, [t, 2 * t])
        if r is not None:
            output.append(r)
    bins.finish(2)
    assert [x[0] for x in output] == [2000.5, 2001.5]
    assert np.allclose([x[1] for x in output], [[2000.5, 4001], [2001.5, 4003]], atol=1e-9, rtol=0)


def test_nonlinear_bin_error_refines_without_right_endpoint_advantage():
    errors = []
    for dt in [0.1, 0.05, 0.025]:
        b = PhysicalBins(dt, 1, 0, [0.0])
        for step in range(1, round(1 / dt) + 1):
            r = b.push(step * dt, [(step * dt) ** 2])
        b.finish(1)
        errors.append(abs(r[1][0] - 1 / 3))
    assert errors[1] == pytest.approx(errors[0] / 4, rel=1e-8)
    assert errors[2] == pytest.approx(errors[1] / 4, rel=1e-8)


@pytest.mark.parametrize(
    "bad", ["shift", "duplicate", "nan", "shape", "partial", "fractional", "start"]
)
def test_invalid_observation_boundaries_refused(bad):
    if bad == "fractional":
        with pytest.raises(ValueError):
            PhysicalBins(0.3, 1, 0, [0])
        return
    if bad == "start":
        with pytest.raises(ValueError):
            PhysicalBins(0.1, 1, 0.2, [0])
        return
    b = PhysicalBins(0.1, 1, 0, [0])
    if bad == "partial":
        b.push(0.1, [0])
        with pytest.raises(ValueError):
            b.finish(1)
    else:
        time = 0.2 if bad == "shift" else 0.1
        value = [np.nan] if bad == "nan" else [0, 0] if bad == "shape" else [0]
        if bad == "duplicate":
            b.push(0.1, [0])
        with pytest.raises(ValueError):
            b.push(time, value)
