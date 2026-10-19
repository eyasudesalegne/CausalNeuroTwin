"""Shared numerical preconditions; HPC-specific CLI/config are not exported."""

import numpy as np

from ..artifacts import write as write


def matrix(a):
    x = np.asarray(a, float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Finite nonnegative square matrix required")
    if not np.allclose(x, x.T, rtol=0, atol=1e-12):
        raise ValueError("Asymmetric matrix")
    return x
