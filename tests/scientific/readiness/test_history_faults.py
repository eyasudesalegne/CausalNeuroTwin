import numpy as np
import pytest

from causalneurotwin.validation.readiness.history_init import (
    HistoryCheckError,
    initialize_history,
)
from causalneurotwin.validation.readiness.wp3 import physical_history

from .test_history_init import simulator


@pytest.mark.parametrize(
    "fault",
    [
        "rolled",
        "cvars",
        "current",
        "delay",
        "weights",
        "coupling",
        "coupling_shape",
        "monitor",
        "negative_delay",
        "float_delay",
        "mode",
        "surface",
        "stimulus",
    ],
)
def test_faults_cannot_pass_quantized_oracles(fault):
    s = simulator()
    if fault in ("rolled", "cvars"):
        original = s.history.initialize

        def corrupt(full):
            original(full)
            s.history.buffer[:] = (
                np.roll(s.history.buffer, 1, axis=0)
                if fault == "rolled"
                else s.history.buffer[:, ::-1]
            )

        s.history.initialize = corrupt
    if fault in ("current", "delay"):
        original = s.history.query_sparse

        def wrong(step):
            now, edge = original(step)
            return (
                (s.history.buffer[(step - 2) % s.history.n_time], edge)
                if fault == "current"
                else (now, original(step + 1)[1])
            )

        s.history.query_sparse = wrong
    if fault == "weights":
        s.history.es_weights[0, 0, 1, 0] = np.nextafter(np.float32(2), np.float32(np.inf))
    if fault == "coupling":

        class Bad:
            def __init__(self, good):
                self.good = good
                self.__dict__.update(good.__dict__)

            def __getattr__(self, k):
                return getattr(self.good, k)

            def __call__(self, *args):
                return self.good(*args) + 1e-6

        s.coupling = Bad(s.coupling)
    if fault == "coupling_shape":

        class BadShape:
            def __init__(self, good):
                self.good = good

            def __getattr__(self, k):
                return getattr(self.good, k)

            def __call__(self, *args):
                return self.good(*args)[:1]

        s.coupling = BadShape(s.coupling)
    if fault == "monitor":
        s.monitors[1].dt = 0.05
    if fault == "negative_delay":
        s.connectivity.idelays[0, 1] = -1
    if fault == "float_delay":
        s.connectivity.idelays = s.connectivity.idelays.astype(float)
    if fault == "mode":
        s.model.cvar = np.array([0, 1])
    if fault == "surface":
        s.surface = object()
    if fault == "stimulus":
        s.stimulus = object()
    diagnostic = {}
    with pytest.raises(HistoryCheckError):
        initialize_history(s, 20, 0.01, 1, diagnostic)
    assert any(c["status"] == "failed" for c in diagnostic["checks"].values())


def test_float32_current_query_is_not_full_float64_state():
    s = simulator()
    r = initialize_history(s, 20, 0.01, 1)
    current, _ = s.history.query_sparse(1)
    assert s.current_state.dtype == np.float64 and current.dtype == np.float32
    assert not np.array_equal(current, s.current_state[[1, 2]])
    assert r["checks"]["current_query"]["passed"]


def test_fractional_asymmetric_weights_match_independent_coupling():
    s = simulator()
    W = np.array(
        [
            [0, 0.123456789, 0.918273645],
            [0.618033989, 0, 0.271828183],
            [0.314159265, 0.141421356, 0],
        ]
    )
    # Update all inspected runtime descriptors and a dense independent summation double.
    s.connectivity.weights = W
    s.history.weights = W
    s.history.es_weights = (
        np.broadcast_to(W[:, None, :, None], (3, 2, 3, 1)).astype(np.float32).copy()
    )
    s.history.nnz_weights = W[W != 0].astype(np.float32)

    class Dense:
        cmin = np.array([0.0])
        cmax = np.array([0.005])
        r = np.array([0.56])
        midpoint = np.array([6.0])
        a = np.array([40.0])

        def __call__(self, step, history):
            _, delayed = history.query_sparse(step)
            out = np.zeros((2, 3, 1))
            k = 0
            for i in range(3):
                for j in range(3):
                    if W[i, j]:
                        value = delayed[0, k, 0] - delayed[1, k, 0]
                        sigmoid = 0.005 / (1 + np.exp(np.float64(0.56) * (np.float64(6) - value)))
                        out[:, i, 0] += np.float64(np.float32(W[i, j])) * sigmoid * 40
                        k += 1
            return out

    s.coupling = Dense()
    assert initialize_history(s, 20, 0.01, 1)["initial_coupling_verified"]


@pytest.mark.parametrize("amplitude", [0, -1, np.nan, np.inf])
def test_invalid_history_amplitude(amplitude):
    with pytest.raises(ValueError):
        physical_history([0.0], 3, 20, amplitude)


@pytest.mark.allocated
def test_native_tvb_smoke_is_explicitly_unexecuted_locally(tmp_path, options):
    import importlib.util

    if importlib.util.find_spec("tvb") is None:
        pytest.skip("TVB absent locally; allocated-node compatibility check remains required")
    from causalneurotwin.validation.readiness.smoke import run_smoke

    assert run_smoke(options, tmp_path)["integration_steps"] == 0
