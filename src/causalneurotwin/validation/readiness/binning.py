"""Matched physical bins from Raw samples with trapezoidal (not endpoint) quadrature."""

import numpy as np


class PhysicalBins:
    def __init__(self, dt, period, start_ms, start_value):
        if not np.isfinite(dt) or not np.isfinite(period) or min(dt, period) <= 0:
            raise ValueError("Positive finite timing required")
        ratio = period / dt
        if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0):
            raise ValueError("Fractional monitor bin")
        if not np.isfinite(start_ms) or not np.isclose(
            start_ms / period, round(start_ms / period), atol=1e-9, rtol=0
        ):
            raise ValueError("Start must be a common physical bin boundary")
        self.dt = float(dt)
        self.period = float(period)
        self.start = float(start_ms)
        self.steps = round(ratio)
        self.count = 0
        self.previous = np.asarray(start_value, dtype=float).copy()
        if not np.isfinite(self.previous).all():
            raise ValueError("Nonfinite starting state")
        self.integral = np.zeros_like(self.previous)

    def push(self, time_ms, value):
        value = np.asarray(value, dtype=float)
        expected = self.start + (self.count + 1) * self.dt
        if not np.isfinite(time_ms) or abs(time_ms - expected) > 1e-8:
            raise ValueError("Missing, repeated, reordered or shifted Raw sample")
        if value.shape != self.previous.shape or not np.isfinite(value).all():
            raise ValueError("Invalid Raw sample")
        self.integral += (self.previous + value) * (self.dt / 2)
        self.previous = value.copy()
        self.count += 1
        if self.count % self.steps:
            return None
        center = self.start + self.count * self.dt - self.period / 2
        average = self.integral / self.period
        self.integral.fill(0)
        return center, average

    def finish(self, duration_ms):
        if (
            not np.isfinite(duration_ms)
            or duration_ms <= 0
            or not np.isclose(self.count * self.dt, duration_ms, atol=1e-8, rtol=0)
            or self.count % self.steps
        ):
            raise ValueError("Incomplete or unexpected physical observation interval")
