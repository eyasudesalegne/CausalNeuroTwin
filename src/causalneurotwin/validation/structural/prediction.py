"""Prespecified held-out metrics only; no EEG loader, fitting or variant selection."""

import numpy as np


def validate_splits(records, heldout_conditions=False):
    if not records:
        raise ValueError("No verified run/condition split supplied")
    required = {"session", "run", "condition", "split"}
    groups = {}
    seen = set()
    condition_splits = {}
    for row in records:
        if not required.issubset(row) or row["split"] not in ("calibration", "validation", "test"):
            raise ValueError("Incomplete split registry")
        if any(
            not isinstance(row[k], str) or not row[k].strip()
            for k in ("session", "run", "condition")
        ):
            raise ValueError("Verified session, run and condition identities required")
        unit = (row["session"], row["run"])
        if unit in groups and groups[unit] != row["split"]:
            raise ValueError("Run leakage across splits")
        groups[unit] = row["split"]
        seen.add(row["split"])
        condition_splits.setdefault(row["condition"], set()).add(row["split"])
    if seen != {"calibration", "validation", "test"}:
        raise ValueError("Independent calibration, validation and test evidence required")
    if heldout_conditions and any(
        "test" in values and len(values) > 1 for values in condition_splits.values()
    ):
        raise ValueError("Claimed held-out condition appears during calibration/validation")
    return {
        "runs": len(groups),
        "heldout_conditions_claimed": heldout_conditions,
        "population_generalization": "not_established by a single participant",
    }


def _corr(a, b, axis):
    a = a - a.mean(axis, keepdims=True)
    b = b - b.mean(axis, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis) * np.sum(b * b, axis))
    return np.divide(
        np.sum(a * b, axis),
        denominator,
        out=np.full(np.shape(denominator), np.nan),
        where=denominator > 1e-15,
    )


def response_metrics(
    predicted,
    measured,
    time_ms,
    window_mask,
    predicted_channels,
    measured_channels,
    units_verified=False,
):
    p = np.asarray(predicted, float)
    y = np.asarray(measured, float)
    t = np.asarray(time_ms, float)
    m = np.asarray(window_mask)
    if p.shape != y.shape or p.ndim != 2 or len(t) != len(y) or p.shape[1] < 2:
        raise ValueError("Identical time/channel layout required")
    if list(predicted_channels) != list(measured_channels) or len(predicted_channels) != p.shape[1]:
        raise ValueError("Channel order mismatch; no silent realignment")
    if (
        m.dtype != bool
        or m.shape != t.shape
        or m.sum() < 3
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Prespecified physical-time/artifact mask required")
    # Masked intervals are ignored; contaminated data are not replaced/interpolated.
    p = p[m]
    y = y[m]
    tt = t[m]
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite evaluated samples")
    scale = max(np.max(abs(p)), np.max(abs(y)), 1e-15)
    if max(abs(p.mean(1)).max(), abs(y.mean(1)).max()) > 1e-10 * scale:
        raise ValueError("Explicit common average reference required")
    temporal = _corr(p, y, 0)
    topography = _corr(p, y, 1)
    valid_temporal = temporal[np.isfinite(temporal)]
    valid_topography = topography[np.isfinite(topography)]
    observed_rms = float(np.sqrt(np.mean(y * y)))
    pgfp = np.sqrt(np.mean(p * p, axis=1))
    ygfp = np.sqrt(np.mean(y * y, axis=1))
    latency = (
        float(tt[np.argmax(pgfp)] - tt[np.argmax(ygfp)])
        if pgfp.max() > 1e-15 and ygfp.max() > 1e-15
        else None
    )
    return {
        "waveform_rmse": float(np.sqrt(np.mean((p - y) ** 2))) if units_verified else None,
        "waveform_nrmse": float(np.sqrt(np.mean((p - y) ** 2)) / observed_rms)
        if units_verified and observed_rms > 1e-15
        else None,
        "temporal_correlation_median": float(np.median(valid_temporal))
        if len(valid_temporal)
        else None,
        "topography_correlation_median": float(np.median(valid_topography))
        if len(valid_topography)
        else None,
        "gfp_peak_latency_error_ms": latency,
        "latency_rule": "earliest maximum within preregistered valid window; no fitted time shift",
        "evaluated_samples": len(tt),
        "evaluated_window_ms": [float(tt.min()), float(tt.max())],
        "amplitude_units_verified": units_verified,
        "fit_performed": False,
    }


def paired_improvement(reference_loss, candidate_loss, reference_units, candidate_units):
    if list(reference_units) != list(candidate_units):
        raise ValueError("Predictions do not share identical held-out units")
    a = np.asarray(reference_loss, float)
    b = np.asarray(candidate_loss, float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or len(a) != len(reference_units)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Finite paired held-out errors required")
    return a - b


def hierarchical_uncertainty(difference, sessions, runs, seed=20261018, draws=2000):
    x = np.asarray(difference, float)
    s = np.asarray(sessions)
    r = np.asarray(runs)
    if (
        x.ndim != 1
        or x.shape != s.shape
        or x.shape != r.shape
        or not np.isfinite(x).all()
        or not isinstance(draws, int)
        or not 50 <= draws <= 10000
    ):
        raise ValueError("Finite paired trial/run/session metadata and bounded bootstrap required")
    # Collapse trials to run means; resample sessions, then their runs.
    levels = {
        session: np.array(
            [x[(s == session) & (r == run)].mean() for run in np.unique(r[s == session])]
        )
        for session in np.unique(s)
    }
    if len(levels) < 2:
        return {
            "status": "indeterminate",
            "reason": "Fewer than two sessions; no independent session uncertainty",
            "population_validity": False,
        }
    rng = np.random.default_rng(seed)
    names = list(levels)
    values = []
    for _ in range(draws):
        sample = []
        for j in rng.integers(0, len(names), len(names)):
            runs_in_session = levels[names[j]]
            sample.append(rng.choice(runs_in_session, len(runs_in_session), replace=True).mean())
        values.append(np.mean(sample))
    return {
        "status": "descriptive_conditional_interval",
        "mean_session_balanced_improvement": float(np.mean([v.mean() for v in levels.values()])),
        "interval95": np.quantile(values, [0.025, 0.975]).tolist(),
        "sessions": len(levels),
        "bootstrap_draws": draws,
        "scope": "single-subject run/session resampling; limited sessions, not population biological validity",
    }
