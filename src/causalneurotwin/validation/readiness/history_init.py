"""TVB 2.10.0: strict float32 storage oracles, independent physical-time indexing."""

import hashlib

import numpy as np

from .wp3 import physical_history


class HistoryCheckError(RuntimeError):
    def __init__(self, name, report):
        self.check = name
        self.report = report
        super().__init__("History verification failed: " + name)


CHECK_NAMES = (
    "supported_configuration",
    "integer_delays",
    "coupling_variables",
    "history_horizon",
    "integrator_bounds",
    "physical_time_zero",
    "buffer_dtype",
    "buffer_order_and_cvars",
    "dense_weight_storage",
    "sparse_weight_storage",
    "sparse_edge_order",
    "current_query",
    "every_active_delay",
    "initial_coupling",
    "monitor_bin_alignment",
)


def slot_times(horizon, dt, current_step=0):
    if (
        not np.isfinite(horizon)
        or int(horizon) != horizon
        or horizon < 1
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid horizon or timestep")
    if not np.isfinite(current_step) or int(current_step) != current_step or current_step < 0:
        raise ValueError("Nonnegative integer current step required")
    return -((int(current_step) - np.arange(int(horizon))) % int(horizon)) * dt


def exact_storage_comparison(actual, reference):
    """Only the correctly rounded float32 representation is admissible: no ULP slack."""
    a = np.asarray(actual)
    r = np.asarray(reference, dtype=np.float64)
    q = r.astype(np.float32)
    shape = a.shape == r.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all() and np.isfinite(q).all())
    passed = shape and finite and a.dtype == np.dtype("float32") and np.array_equal(a, q)
    out = {
        "passed": bool(passed),
        "actual_dtype": str(a.dtype),
        "reference_dtype": str(r.dtype),
        "expected_storage_dtype": "float32",
        "actual_shape": list(a.shape),
        "expected_shape": list(r.shape),
        "rule": "exact equality to reference.astype(float32); zero extra tolerance",
        "reference_quantization_max_abs": float(np.max(abs(q.astype(float) - r)))
        if r.size
        else 0.0,
    }
    if shape and finite:
        diff = a.astype(float) - r
        out.update(
            max_abs_error_to_float64=float(abs(diff).max()) if a.size else 0.0,
            max_abs_error_to_expected_float32=float(abs(a.astype(float) - q.astype(float)).max())
            if a.size
            else 0.0,
            mismatched_elements=int(np.count_nonzero(a != q)),
            mismatch_examples=[
                {
                    "index": idx.tolist(),
                    "actual": float(a[tuple(idx)]),
                    "expected_float32": float(q[tuple(idx)]),
                    "reference_float64": float(r[tuple(idx)]),
                }
                for idx in np.argwhere(a != q)[:8]
            ],
        )
    return out


def check_storage(name, actual, reference, report):
    detail = exact_storage_comparison(actual, reference)
    report["checks"][name] = {"status": "passed" if detail["passed"] else "failed", **detail}
    if not detail["passed"]:
        raise HistoryCheckError(name, report)
    return detail


def gate(name, condition, report, **detail):
    report["checks"][name] = {"status": "passed" if condition else "failed", **detail}
    if not condition:
        raise HistoryCheckError(name, report)


def coupling_comparison(actual, expected, terms):
    """Float64 summation error budget after exact float32 storage checks.
    gamma_k bounds alternative addition orders; safety factor covers subtraction/exp
    and multiply/add roundoff. This is not a float32 tolerance on history ordering.
    """
    a = np.asarray(actual)
    e = np.asarray(expected, dtype=float)
    k = 4 * max(1, int(terms)) + 32
    eps = np.finfo(np.float64).eps
    gamma = k * eps / (1 - k * eps)
    limit = gamma * abs(e) + np.finfo(np.float64).tiny
    valid = a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
    ok = valid and bool(np.all(abs(a - e) <= limit))
    return {
        "passed": bool(ok),
        "actual_dtype": str(a.dtype),
        "expected_dtype": str(e.dtype),
        "gamma_roundoff": float(gamma),
        "terms_bound": int(terms),
        "max_abs_error": float(abs(a - e).max()) if valid else None,
        "max_abs_error_limit": float(limit.max()) if valid else None,
    }


def verify_queries(sim, step, seed, amplitude, report, prefix=""):
    """Chronological scalar-edge oracle independent of flattened buffer indexing."""
    report["query_step"] = int(step)
    dt = float(sim.integrator.dt)
    n = int(sim.connectivity.number_of_regions)
    target, source = np.nonzero(sim.connectivity.weights)
    current, delayed = sim.history.query_sparse(step)
    t = (int(step) - 1) * dt
    now = physical_history([t], n, seed, amplitude)[0, [1, 2]]
    check_storage(prefix + "current_query", current, now, report)
    lag = sim.connectivity.idelays[target, source]
    times = (int(step) - 1 - lag) * dt
    # Each edge is evaluated at its source/time, without consulting the runtime buffer.
    unique, inverse = np.unique(times, return_inverse=True)
    chronological = physical_history(unique, n, seed, amplitude)
    expected = np.empty((2, len(source), 1), dtype=float)
    for edge, node in enumerate(source):
        expected[:, edge, :] = chronological[inverse[edge], [1, 2], node, :]
    check_storage(prefix + "every_active_delay", delayed, expected, report)
    return expected.astype(np.float32)


def explicit_coupling(sim, delayed):
    """Independent active-edge accumulation using inspected dense-history float32 weights."""
    n = int(sim.connectivity.number_of_regions)
    target, source = np.nonzero(sim.connectivity.weights)
    coup = sim.coupling
    # Subtraction happens in float32 in installed Coupling.pre, before float64 parameters.
    signal = delayed[0] - delayed[1]
    transformed = np.asarray(coup.cmin) + (np.asarray(coup.cmax) - np.asarray(coup.cmin)) / (
        1 + np.exp(np.asarray(coup.r) * (np.asarray(coup.midpoint) - signal))
    )
    weights = np.asarray(sim.connectivity.weights, dtype=np.float32)
    result = np.zeros((1, n, 1), dtype=float)
    for k, i in enumerate(target):
        result[0, i] += float(weights[i, source[k]]) * transformed[k]
    return np.asarray(coup.a) * np.repeat(result, 2, axis=0)


def initialize_history(sim, seed, amplitude, monitor_ms, report=None):
    report = {} if report is None else report
    report.update(
        physical_origin_ms=0.0,
        current_step=0,
        indexing="query(s): (s-1-delay)%horizon; update(s): s%horizon",
        checks={name: {"status": "not_run"} for name in CHECK_NAMES},
    )
    period = float(monitor_ms)
    dt = float(sim.integrator.dt)
    supported = (
        sim.surface is None
        and sim.stimulus is None
        and sim.calls == 0
        and np.isfinite(dt)
        and dt > 0
        and np.isfinite(period)
        and period > 0
        and np.isclose(period / dt, round(period / dt), atol=1e-9, rtol=0)
    )
    gate("supported_configuration", supported, report)
    n = int(sim.connectivity.number_of_regions)
    delays = np.asarray(sim.connectivity.idelays)
    gate(
        "integer_delays",
        delays.shape == (n, n)
        and np.issubdtype(delays.dtype, np.integer)
        and bool(np.all(delays >= 0)),
        report,
    )
    cvars = np.asarray(sim.model.cvar)
    gate(
        "coupling_variables",
        np.array_equal(cvars, [1, 2]) and sim.model.nvar == 6 and sim.model.number_of_modes == 1,
        report,
        selected_cvars=cvars.tolist(),
    )
    horizon = int(delays.max()) + 1
    gate("history_horizon", sim.history.n_time == horizon, report, horizon=horizon)
    times = slot_times(horizon, dt)
    full = physical_history(times, n, seed, amplitude)
    bounded = full.copy()
    for state in bounded:
        sim.integrator.bound_and_clamp(state)
    gate("integrator_bounds", np.array_equal(bounded, full), report)
    sim.history.initialize(full.copy())
    sim.current_step = 0
    sim.current_state = full[0].copy()
    expected_zero = physical_history([0.0], n, seed, amplitude)[0]
    gate(
        "physical_time_zero",
        sim.current_state.dtype == np.dtype("float64")
        and np.array_equal(sim.current_state, expected_zero),
        report,
        current_state_dtype=str(sim.current_state.dtype),
    )
    gate(
        "buffer_dtype",
        sim.history.buffer.dtype == np.dtype("float32"),
        report,
        dtype=str(sim.history.buffer.dtype),
    )
    check_storage("buffer_order_and_cvars", sim.history.buffer, full[:, [1, 2]], report)
    W = np.asarray(sim.connectivity.weights)
    target, source = np.nonzero(W)
    gate("active_edges", len(source) > 0 and np.isfinite(W).all() and bool(np.all(W >= 0)), report)
    # The dense coupling path uses es_weights; sparse descriptor also casts its weights.
    check_storage(
        "dense_weight_storage",
        sim.history.es_weights,
        np.broadcast_to(W[:, None, :, None], (n, 2, n, 1)),
        report,
    )
    check_storage("sparse_weight_storage", sim.history.nnz_weights, W[target, source], report)
    gate(
        "sparse_edge_order",
        np.array_equal(sim.history.nnz_row_el_idx, target)
        and np.array_equal(sim.history.nnz_col_el_idx, source)
        and np.array_equal(sim.history.nnz_idelays, delays[target, source]),
        report,
    )
    expected = verify_queries(sim, 1, seed, amplitude, report)
    oracle = explicit_coupling(sim, expected)
    actual = sim.coupling(1, sim.history)
    detail = coupling_comparison(actual, oracle, n)
    gate(
        "initial_coupling",
        detail["passed"],
        report,
        **{k: v for k, v in detail.items() if k != "passed"},
    )
    mon = sim.monitors[1] if len(sim.monitors) == 2 else None
    expected_stock = (round(period / dt), 6, n, 1)
    aligned = (
        mon is not None
        and mon.istep == round(period / dt)
        and np.shape(mon._stock) == expected_stock
        and np.isfinite(mon._stock).all()
        and bool(np.all(mon._stock == 0))
        and np.array_equal(mon.voi, np.arange(6))
        and float(mon.period) == period
        and float(mon.dt) == dt
    )
    gate(
        "monitor_bin_alignment",
        aligned,
        report,
        period_ms=period,
        first_native_bin_center_ms=period / 2,
        first_native_bin_end_ms=period,
        note="Native TemporalAverage uses right-endpoint samples; intrinsic accuracy uses matched trapezoidal Raw bins.",
    )
    report.update(
        horizon=horizon,
        history_time_min_ms=float(times.min()),
        history_time_max_ms=0.0,
        monitor_period_ms=period,
        monitor_boundary_aligned=True,
        active_delayed_edges_verified=len(source),
        initial_coupling_verified=True,
        current_state_sha256=hashlib.sha256(sim.current_state.tobytes()).hexdigest(),
        buffer_sha256=hashlib.sha256(sim.history.buffer.tobytes()).hexdigest(),
        method="Exact physical history; exact expected float32 storage; independent edge/coupling oracles",
    )
    return report
