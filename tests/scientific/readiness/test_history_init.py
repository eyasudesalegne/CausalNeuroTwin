"""Nonparticipant buffer oracle; actual TVB remains an allocated-node requirement."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from causalneurotwin.validation.readiness.history_init import initialize_history, slot_times
from causalneurotwin.validation.readiness.wp3 import physical_history


class History:
    def __init__(self, delays, weights):
        self.delays = delays
        self.weights = weights
        self.n_time = int(delays.max()) + 1
        self.es_weights = (
            np.broadcast_to(weights[:, None, :, None], (3, 2, 3, 1)).astype(np.float32).copy()
        )
        self.nnz_row_el_idx, self.nnz_col_el_idx = np.nonzero(weights)
        self.nnz_weights = weights[weights != 0].astype(np.float32)
        self.nnz_idelays = delays[weights != 0]

    def initialize(self, full):
        self.buffer = full[:, [1, 2]].astype(np.float32)

    def query_sparse(self, step):
        target, source = np.nonzero(self.weights)
        at = (step - 1 - self.delays[target, source]) % self.n_time
        # Native TVB flattened take: result axes are cvar, edge, mode.
        const = np.arange(2)[:, None, None] * 3 + source[None, :, None]
        edge = self.buffer.take(at[:, None] * 6 + const)
        return self.buffer[(step - 1) % self.n_time], edge

    def update(self, step, state):
        self.buffer[step % self.n_time] = state[[1, 2]]


def simulator(dt=0.1):
    W = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0.0]])
    D = np.rint(np.array([[0, 0.53, 0.97], [0.53, 0, 0.29], [0.97, 0.29, 0]]) / dt).astype(int)
    hist = History(D, W)

    class Coupling:
        cmin = np.array([0.0])
        cmax = np.array([0.005])
        r = np.array([0.56])
        midpoint = np.array([6.0])
        a = np.array([40.0])

        def __call__(self, step, history):
            _, edge = history.query_sparse(step)
            out = np.zeros((2, 3, 1))
            # Independent edge loop rather than the implementation's reduction.
            for k, (i, j) in enumerate(zip(*np.nonzero(W))):
                signal = edge[0, k, 0] - edge[1, k, 0]
                # Match documented storage, but use an independent scalar sum.
                term = np.float64(0.005) / (1 + np.exp(np.float64(0.56) * (np.float64(6) - signal)))
                out[:, i, 0] += 40 * float(np.float32(W[i, j])) * term
            return out

    return NS(
        surface=None,
        stimulus=None,
        calls=0,
        current_step=99,
        current_state=None,
        history=hist,
        coupling=Coupling(),
        connectivity=NS(idelays=D, weights=W, number_of_regions=3),
        model=NS(cvar=np.array([1, 2]), nvar=6, number_of_modes=1),
        integrator=NS(dt=dt, bound_and_clamp=lambda x: None),
        monitors=[
            NS(),
            NS(
                istep=round(1 / dt),
                period=1.0,
                dt=dt,
                voi=np.arange(6),
                _stock=np.zeros((round(1 / dt), 6, 3, 1)),
            ),
        ],
    )


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
def test_exact_current_delays_coupling_and_monitor_boundaries(dt):
    sim = simulator(dt)
    r = initialize_history(sim, 20, 0.01, 1)
    assert np.array_equal(sim.current_state, physical_history([0.0], 3, 20)[0])
    assert sim.current_step == 0 and sim.calls == 0
    assert r["active_delayed_edges_verified"] == 6 and r["initial_coupling_verified"]


def test_common_physical_samples_and_current_states_across_dt():
    sims = [simulator(dt) for dt in (0.1, 0.05, 0.025)]
    for s in sims:
        initialize_history(s, 20, 0.01, 1)
    assert all(np.array_equal(s.current_state, sims[0].current_state) for s in sims)
    for lag in (0, 0.1, 0.2, 0.3):
        values = [
            s.history.buffer[(-round(lag / s.integrator.dt)) % s.history.n_time] for s in sims
        ]
        assert all(np.allclose(x, values[0], atol=1e-15, rtol=0) for x in values)


@pytest.mark.parametrize("horizon,current", [(1, 0), (7, 0), (7, 3), (7, 21)])
def test_slot_layout_handles_wrapping(horizon, current):
    times = slot_times(horizon, 0.1, current)
    for lag in range(horizon):
        assert times[(current - lag) % horizon] == pytest.approx(-lag * 0.1)


def test_updates_and_delayed_queries_across_multiple_wraps():
    sim = simulator()
    initialize_history(sim, 20, 0.01, 1)
    h = sim.history
    for step in range(1, 3 * h.n_time + 1):
        current, edge = h.query_sparse(step)
        assert np.allclose(
            current,
            physical_history([(step - 1) * 0.1], 3, 20)[0, [1, 2]].astype(np.float32),
            atol=0,
            rtol=0,
        )
        for k, (target, source) in enumerate(zip(*np.nonzero(h.weights))):
            t = (step - 1 - h.delays[target, source]) * 0.1
            assert np.allclose(
                edge[:, k],
                physical_history([t], 3, 20)[0, [1, 2], source].astype(np.float32),
                atol=0,
                rtol=0,
            )
        h.update(step, physical_history([step * 0.1], 3, 20)[0])


@pytest.mark.parametrize("bad", ["bounds", "monitor", "used", "query", "coupling", "horizon"])
def test_unestablished_initialization_fails_explicitly(bad):
    s = simulator()
    if bad == "bounds":
        s.integrator.bound_and_clamp = lambda x: x.fill(0)
    if bad == "monitor":
        s.monitors[1]._stock.fill(1)
    if bad == "used":
        s.calls = 1
    if bad == "query":
        s.history.query_sparse = lambda step: (np.zeros((2, 3, 1)), np.zeros((2, 6, 1)))
    if bad == "coupling":
        s.coupling.a = np.array([np.nan])
    if bad == "horizon":
        s.history.n_time += 1
    with pytest.raises(RuntimeError):
        initialize_history(s, 20, 0.01, 1)


def test_native_sparse_axis_contract_is_cvar_edge_mode():
    s = simulator()
    initialize_history(s, 20, 0.01, 1)
    _, edges = s.history.query_sparse(1)
    assert edges.shape == (2, 6, 1)
    assert not np.array_equal(edges[0], edges[1])
    assert not np.array_equal(edges[:, 0], edges[:, 1])


def test_bad_clock_and_layout_are_refused():
    with pytest.raises(ValueError):
        slot_times(0, 0.1)
    with pytest.raises(ValueError):
        slot_times(3, float("nan"))
    with pytest.raises(ValueError):
        slot_times(3, 0.1, 0.5)
    s = simulator()
    with pytest.raises(RuntimeError):
        initialize_history(s, 20, 0.01, 0.33)
