"""Numerical acceptance and provenance interpretation; no biological calibration."""

import numpy as np


def history_grid(max_delay_ms, dt_ms, monitor_ms):
    if (
        not all(np.isfinite([max_delay_ms, dt_ms, monitor_ms]))
        or min(dt_ms, monitor_ms) <= 0
        or max_delay_ms < 0
    ):
        raise ValueError("Invalid history grid")
    if not np.isclose(monitor_ms / dt_ms, round(monitor_ms / dt_ms), atol=1e-9, rtol=0):
        raise ValueError("Monitor period must contain integer integration steps")
    # Start all conditions at the SAME full monitor-bin boundary, with enough delay history.
    duration = (np.ceil(max_delay_ms / monitor_ms) + 1) * monitor_ms
    count = round(duration / dt_ms)
    return (np.arange(count + 1) - count) * dt_ms


def physical_history(times_ms, nodes, seed, amplitude=0.01):
    t = np.asarray(times_ms, float)
    if (
        t.ndim != 1
        or not np.isfinite(t).all()
        or not np.isfinite(amplitude)
        or amplitude <= 0
        or int(nodes) != nodes
        or nodes < 1
    ):
        raise ValueError("Invalid physical history")
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (6, nodes, 1))
    # Same physical function across dt; no reuse/interpolation of an incompatible delay buffer.
    return amplitude * np.sin(2 * np.pi * 8 * t[:, None, None, None] / 1000 + phase[None])


def source_features(time_ms, source):
    t = np.asarray(time_ms, float)
    x = np.asarray(source, float)
    if (
        x.ndim != 2
        or len(t) != len(x)
        or len(t) < 32
        or not np.isfinite(x).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Finite time-by-region samples required")
    delta = np.diff(t)
    if not np.allclose(delta, delta[0], atol=1e-8, rtol=0) or delta[0] <= 0:
        raise ValueError("Nonuniform time grid")
    freq = np.fft.rfftfreq(len(t), delta[0] / 1000)
    y = x - x.mean(0)
    window = np.hanning(len(t))[:, None]
    psd = abs(np.fft.rfft(y * window, axis=0)) ** 2 / (window[:, 0] @ window[:, 0])
    band = (freq >= 1) & (freq <= 40)
    if not band.any():
        raise ValueError("Insufficient spectral window")
    return {
        "std": x.std(0),
        "peak_hz": freq[band][psd[band].argmax(0)],
        "bandpower": psd[band].sum(0),
        "mean": x.mean(0),
    }


def compare_runs(reference, candidate, thresholds, require_trajectory=True):
    t, x = reference
    t2, y = candidate
    t, t2 = np.asarray(t), np.asarray(t2)
    if t.shape != t2.shape or not np.allclose(t, t2, atol=1e-6, rtol=0):
        raise ValueError("Common physical monitor grid required")
    if np.shape(x) != np.shape(y):
        raise ValueError("Region layout mismatch")
    a, b = source_features(t, x), source_features(t2, y)
    floor = 1e-12
    if np.any(a["std"] <= floor) or np.any(a["bandpower"] <= floor):
        return {
            "status": "indeterminate",
            "reason": "Reference has flat/unresolved regional dynamics",
        }
    nrmse = float(
        np.sqrt(np.mean((np.asarray(x) - y) ** 2))
        / np.sqrt(np.mean((np.asarray(x) - np.mean(x, axis=0)) ** 2))
    )
    std = float(np.quantile(abs(b["std"] - a["std"]) / a["std"], 0.95))
    peak = float(np.quantile(abs(b["peak_hz"] - a["peak_hz"]), 0.95))
    power = float(np.quantile(abs(b["bandpower"] - a["bandpower"]) / a["bandpower"], 0.95))
    regional = np.sqrt(np.mean((np.asarray(x) - y) ** 2, axis=0)) / a["std"]
    metrics = {
        "regional_nrmse_p95": float(np.quantile(regional, 0.95)),
        "nrmse": nrmse,
        "relative_std": std,
        "peak_frequency_hz": peak,
        "relative_bandpower": power,
    }
    relevant = (
        metrics
        if require_trajectory
        else {k: v for k, v in metrics.items() if k not in ("nrmse", "regional_nrmse_p95")}
    )
    passed = all(
        v <= thresholds["nrmse" if k == "regional_nrmse_p95" else k] for k, v in relevant.items()
    )
    return {
        "status": "numerical_pass" if passed else "numerical_fail",
        "metrics": metrics,
        "trajectory_required": require_trajectory,
        "thresholds_provisional": thresholds,
        "independent_biological_validation": "not_established",
    }


def condition_plan(options):
    a, b, c = options["seeds"]
    dt = options["dt_ms"]
    if dt != [0.1, 0.05, 0.025] or options["warmups_ms"] != [1000, 2000, 4000]:
        raise ValueError("Unreviewed numerical design")
    return [
        {"name": "dt_0p1", "dt": dt[0], "warm": 2000, "seed": a, "amplitude": 0.01},
        {"name": "dt_0p05", "dt": dt[1], "warm": 2000, "seed": a, "amplitude": 0.01},
        {"name": "dt_0p025", "dt": dt[2], "warm": 2000, "seed": a, "amplitude": 0.01},
        {"name": "initial_seed_B", "dt": dt[1], "warm": 2000, "seed": b, "amplitude": 0.01},
        {"name": "initial_seed_C", "dt": dt[1], "warm": 2000, "seed": c, "amplitude": 0.01},
        {"name": "warm_1000", "dt": dt[1], "warm": 1000, "seed": a, "amplitude": 0.01},
        {"name": "warm_4000", "dt": dt[1], "warm": 4000, "seed": a, "amplitude": 0.01},
        {"name": "initial_amplitude", "dt": dt[1], "warm": 2000, "seed": a, "amplitude": 0.1},
    ]


def numerical_assessment(comparisons, phases):
    failed = [key for key, r in comparisons.items() if r["status"] != "numerical_pass"]
    incomplete = not comparisons or not phases
    nonfinite = any(p.get("nonfinite_samples", 0) > 0 for p in phases)
    warnings = sum(p.get("runtime_warning_count", 0) for p in phases)
    overflow_exponents = sum(p.get("evaluations", {}).get("overflow_exponents", 0) for p in phases)
    accuracy_failed = [k for k in failed if not k.endswith("_statistics")]
    robustness_failed = [k for k in failed if k.endswith("_statistics")]
    return {
        "status": "numerical_fail"
        if accuracy_failed or nonfinite
        else "indeterminate"
        if incomplete or warnings or overflow_exponents
        else "robustness_sensitive"
        if robustness_failed
        else "numerical_pass",
        "accuracy_failed": accuracy_failed,
        "robustness_sensitive_comparisons": robustness_failed,
        "failed_comparisons": failed,
        "incomplete_evidence": incomplete,
        "runtime_warnings": warnings,
        "evaluated_overflow_exponents": overflow_exponents,
        "nonfinite_detected": nonfinite,
        "warning_policy": "No overflow suppression, clipping or automatic parameter adjustment",
        "biological_calibration": "not_performed",
        "TMS_response_convergence": "not_tested",
    }


def null_claims(single, ensemble, member):
    # Reuse supplied provenances, distinguish construction guarantees from independent recalculation.
    error = float(single["node_weighted_strength_max_relative_error"])
    tolerance = float(single["node_weighted_strength_tolerance"])
    contradictory = bool(
        single["global_original_weight_distribution_preserved"]
        or single["weight_length_bundle_distribution_preserved"]
        or member["weight_distribution_preserved_after_strength_balance"]
        or member["weight_length_bundle_distribution_preserved_after_balance"]
    )
    claims = {
        "binary_degree": "preserved",
        "node_strength": f"preserved to {tolerance:g} tolerance; not bitwise identity",
        "global_length_distribution": "preserved in ensemble",
        "endpoint_specific_delays": "not_preserved",
        "original_weight_distribution": "not_preserved",
        "original_weight_length_pairings": "not_preserved",
        "pure_topology_intervention": False,
    }
    return {
        "claims": claims,
        "single_max_relative_strength_error": error,
        "ensemble_max_relative_strength_error": ensemble[
            "maximum_strength_relative_error_across_ensemble"
        ],
        "provenance_consistent": not contradictory
        and error < tolerance
        and bool(single["degree_sequence_preserved_exactly"])
        and bool(ensemble["degree_sequence_preserved"])
        and bool(ensemble["tract_length_distribution_preserved_exactly_per_member"]),
        "verification_scope": "Read saved construction reports and reviewed source; no nulls rebuilt or experiments rerun",
        "anatomical_validation": "not_established",
    }


def overflow_evidence(baseline_log, continuation_log):
    lines = baseline_log.splitlines()
    overflow = [x for x in lines if "overflow" in x.lower()]
    import re

    warm = re.search(r"Warm-up warnings\s*:\s*(\d+)", continuation_log)
    cont = re.search(r"Continuation warnings\s*:\s*(\d+)", continuation_log)
    return {
        "saved_overflow_messages": overflow,
        "warmup_warnings": int(warm[1]) if warm else None,
        "continuation_warnings": int(cont[1]) if cont else None,
        "interpretation": "Finite output does not rule out intermediate overflow; production cleanliness does not validate initial-history robustness",
        "mechanism": "Installed JR and coupling sources use exponentials; saturation is plausible, not proven by these logs",
        "G40_warmup_policy": "Existing G40 t0 v2 source uses np.errstate(over='ignore', invalid='ignore'); warning absence there is not evidence of no overflow",
    }


def refinement_assessment(coarse, fine, criteria):
    """Observed refinement gate, not a claim of pure Heun order with rounded delays."""
    if "metrics" not in coarse or "metrics" not in fine:
        return {"status": "indeterminate", "reason": "Unresolved dynamics"}
    a = coarse["metrics"]["regional_nrmse_p95"]
    b = fine["metrics"]["regional_nrmse_p95"]
    limit = a * (1 + criteria["relative_slack"]) + criteria["absolute_floor"]
    return {
        "status": "numerical_pass" if b <= limit else "numerical_fail",
        "coarse_pair_p95_nrmse": a,
        "fine_pair_p95_nrmse": b,
        "limit": limit,
        "thresholds_provisional": criteria,
        "interpretation": "Combined timestep, native rounded-delay and float32-history refinement; not physiological validation",
    }
