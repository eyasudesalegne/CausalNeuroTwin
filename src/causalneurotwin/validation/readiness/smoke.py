"""Allocated-node SYNTHETIC configuration/history/monitor check: zero ODE steps."""

import numpy as np

from .common import write
from .history_init import (
    HistoryCheckError,
    check_storage,
    gate,
    initialize_history,
    slot_times,
    verify_queries,
)
from .runtime import tvb_runtime
from .wp3 import condition_plan, history_grid, physical_history


def monitor_probe(sim, seed, amplitude, period, report):
    """Exercise the real monitor with supplied synthetic states, never an integrator."""
    mon = sim.monitors[1]
    dt = float(sim.integrator.dt)
    n = sim.connectivity.number_of_regions
    count = round(period / dt)
    stock = []
    for step in range(1, 2 * count + 1):
        state = physical_history([step * dt], n, seed, amplitude)[0]
        stock.append(state)
        actual = mon.sample(step, state)
        if step % count:
            gate("monitor_no_early_bin_" + str(step), actual is None, report)
        else:
            expected = np.mean(np.stack(stock[-count:]), axis=0)
            center = (step - count / 2) * dt
            ok = (
                actual is not None
                and abs(float(actual[0]) - center) < 1e-12
                and np.array_equal(actual[1], expected)
            )
            gate(
                "monitor_boundary_" + str(step),
                ok,
                report,
                physical_center_ms=center,
                physical_bin_end_ms=step * dt,
            )
    mon._stock.fill(0)


def run_smoke(c, out):
    reports = []
    states = {}
    sim = None
    last_update = 0
    result = {
        "status": "not_completed",
        "conditions": reports,
        "simulation_executed": False,
        "integration_steps": 0,
        "participant_arrays_loaded": False,
    }
    try:
        Connectivity, simulator, models, coupling, integrators, monitors = tvb_runtime(c, out)
        opt = c["wp3"]
        # Directed, nonbinary, non-float32-exact weights expose source/target and storage faults.
        W = np.array(
            [[0, 0.123456789, 0], [0.618033989, 0, 0.271828183], [0.314159265, 0.141421356, 0]]
        )
        L = np.array([[0, 5.3, 0], [7.1, 0, 2.9], [9.7, 3.1, 0]])
        C = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0.0]], float)
        for condition in condition_plan(opt):
            item = {"condition": condition, "checks": {}, "integration_steps": 0}
            reports.append(item)
            dt = condition["dt"]
            last_update = 0
            sim = None
            conn = Connectivity(
                weights=W.copy(),
                tract_lengths=L.copy(),
                centres=C.copy(),
                region_labels=np.array(["S0", "S1", "S2"]),
                speed=np.array([opt["speed_mm_ms"]]),
            )
            conn.configure()
            model = models.JansenRit()
            model.variables_of_interest = list(model.state_variables)
            initial = physical_history(
                history_grid(float((L / opt["speed_mm_ms"]).max()), dt, opt["monitor_ms"]),
                3,
                condition["seed"],
                condition["amplitude"],
            )
            sim = simulator.Simulator(
                model=model,
                connectivity=conn,
                coupling=coupling.SigmoidalJansenRit(a=np.array([opt["gain"]])),
                integrator=integrators.HeunDeterministic(dt=dt),
                monitors=(monitors.Raw(), monitors.TemporalAverage(period=opt["monitor_ms"])),
                initial_conditions=initial,
            )
            sim.configure()
            initialize_history(
                sim, condition["seed"], condition["amplitude"], opt["monitor_ms"], item
            )
            states[condition["name"]] = sim.current_state.copy()
            # Prove corruption is rejected, then restore exact copied buffer; never relax tolerances.
            saved = sim.history.buffer.copy()
            for fault in ("slot_roll", "coupling_swap", "one_ulp"):
                try:
                    if fault == "slot_roll":
                        sim.history.buffer[:] = np.roll(saved, 1, axis=0)
                    elif fault == "coupling_swap":
                        sim.history.buffer[:] = saved[:, ::-1]
                    else:
                        sim.history.buffer[:] = saved
                        sim.history.buffer[0, 0, 0, 0] = np.nextafter(
                            saved[0, 0, 0, 0], np.float32(np.inf)
                        )
                    negative = {}
                    negative["checks"] = {}
                    reference = physical_history(
                        slot_times(sim.history.n_time, dt),
                        3,
                        condition["seed"],
                        condition["amplitude"],
                    )[:, [1, 2]]
                    try:
                        check_storage("injected_fault", sim.history.buffer, reference, negative)
                    except HistoryCheckError:
                        detected = True
                    else:
                        detected = False
                    gate("reject_" + fault, detected, item)
                finally:
                    sim.history.buffer[:] = saved
            original_pre = sim.coupling.pre
            try:
                sim.coupling.pre = lambda a, b, pre=original_pre: pre(a, b) + 0.001
                try:
                    initialize_history(
                        sim, condition["seed"], condition["amplitude"], opt["monitor_ms"], {}
                    )
                except HistoryCheckError as error:
                    detected = error.check == "initial_coupling"
                else:
                    detected = False
            finally:
                sim.coupling.pre = original_pre
            gate("reject_wrong_coupling", detected, item)
            sim.calls = 1
            try:
                try:
                    initialize_history(
                        sim, condition["seed"], condition["amplitude"], opt["monitor_ms"], {}
                    )
                except HistoryCheckError as error:
                    detected = error.check == "supported_configuration"
                else:
                    detected = False
            finally:
                sim.calls = 0
            gate("reject_used_simulator", detected, item)
            # All active edges at each synthetic update across three complete buffer wraps.
            for step in range(1, 3 * sim.history.n_time + 1):
                verify_queries(
                    sim, step, condition["seed"], condition["amplitude"], item, prefix="wrap_"
                )
                sim.history.update(
                    step,
                    physical_history([step * dt], 3, condition["seed"], condition["amplitude"])[0],
                )
                last_update = step
            item["synthetic_history_updates"] = 3 * sim.history.n_time
            gate(
                "history_wrapping",
                True,
                item,
                queries_checked=3 * sim.history.n_time,
                active_edges_per_query=int(np.count_nonzero(W)),
                note="Every query checked; detailed fields show final query.",
            )
            monitor_probe(sim, condition["seed"], condition["amplitude"], opt["monitor_ms"], item)
            initialize_report = {}
            initialize_history(
                sim, condition["seed"], condition["amplitude"], opt["monitor_ms"], initialize_report
            )
            gate("zero_integration", sim.current_step == 0 and sim.calls == 0, item)
        for name in ("dt_0p025", "dt_0p1", "warm_1000", "warm_4000"):
            gate(
                "common_state_" + name, np.array_equal(states[name], states["dt_0p05"]), reports[0]
            )
        result.update(
            status="configuration_checks_passed",
            biological_validation="not_established",
            full_intrinsic_execution="not_performed",
        )
        write(out / "history_smoke.json", result)
        return {
            "runtime_configuration": "passed",
            "integration_steps": 0,
            "biological_validation": "not_established",
            "full_intrinsic_execution": "not_performed",
        }
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, reason=str(error))
        try:
            if sim is not None and reports and getattr(sim, "history", None) is not None:
                item = reports[-1]
                dt = float(sim.integrator.dt)
                reference = physical_history(
                    slot_times(sim.history.n_time, dt),
                    3,
                    item["condition"]["seed"],
                    item["condition"]["amplitude"],
                )
                with (out / "synthetic_failure_arrays.npz").open("xb") as h:
                    np.savez_compressed(
                        h,
                        actual_buffer=sim.history.buffer,
                        reference_float64=reference,
                        expected_float32=reference[:, [1, 2]].astype(np.float32),
                        reference_times_at_last_update=last_update * dt
                        + slot_times(sim.history.n_time, dt, last_update),
                        current_state=sim.current_state,
                        integer_delays=sim.connectivity.idelays,
                        weights=sim.connectivity.weights,
                    )
                result["failure_arrays"] = "synthetic_failure_arrays.npz"
            else:
                result["failure_arrays"] = "unavailable_before_history_configuration"
        except Exception as diagnostic_error:
            result["diagnostic_capture_error"] = str(diagnostic_error)
        write(out / "history_smoke.json", result)
        raise
