"""Observe evaluated RHS/predictor and delayed-coupling arguments without changing them."""

import numpy as np

from .common import write


class EvaluationAudit:
    def __init__(self, model, coupling, out):
        self.model = model
        self.coupling = coupling
        self.out = out
        self.begin("configuration")
        self.original_dfun = model.dfun
        self.original_pre = coupling.pre
        model.dfun = self.dfun
        coupling.pre = self.pre

    def begin(self, phase):
        self.data = {
            "phase": phase,
            "rhs_calls": 0,
            "coupling_pre_calls": 0,
            "max_sigmoid_exponent": None,
            "overflow_exponents": 0,
            "max_abs_evaluated_state": 0.0,
        }

    def snapshot(self):
        return dict(self.data)

    def finite(self, value, stage):
        if not np.isfinite(value).all():
            p = self.out / "numerical_failure.json"
            if not p.exists():
                write(p, {"stage": stage, "evaluation_audit": self.snapshot()})
            raise FloatingPointError("Nonfinite " + stage)

    def exponents(self, args):
        for value in args:
            self.finite(value, "sigmoid_argument")
            maximum = float(np.max(value))
            self.data["max_sigmoid_exponent"] = (
                maximum
                if self.data["max_sigmoid_exponent"] is None
                else max(maximum, self.data["max_sigmoid_exponent"])
            )
            self.data["overflow_exponents"] += int(
                np.count_nonzero(value > np.log(np.finfo(np.float64).max))
            )

    def dfun(self, state, *args, **kwargs):
        self.data["rhs_calls"] += 1
        self.finite(state, "RHS_input")
        y = np.asarray(state)[..., 0]
        m = self.model
        self.data["max_abs_evaluated_state"] = max(
            self.data["max_abs_evaluated_state"], float(abs(y).max())
        )
        scalar = lambda key: float(np.asarray(getattr(m, key)).ravel()[0])
        self.exponents(
            [
                scalar("r") * (scalar("v0") - (y[1] - y[2])),
                scalar("r") * (scalar("v0") - scalar("a_1") * scalar("J") * y[0]),
                scalar("r") * (scalar("v0") - scalar("a_3") * scalar("J") * y[0]),
            ]
        )
        result = self.original_dfun(state, *args, **kwargs)
        self.finite(result, "RHS_derivative")
        return result

    def pre(self, x_i, x_j):
        self.data["coupling_pre_calls"] += 1
        self.finite(x_j, "delayed_coupling_state")
        c = self.coupling
        self.exponents([np.asarray(c.r) * (np.asarray(c.midpoint) - (x_j[:, 0] - x_j[:, 1]))])
        result = self.original_pre(x_i, x_j)
        self.finite(result, "coupling_output")
        return result
