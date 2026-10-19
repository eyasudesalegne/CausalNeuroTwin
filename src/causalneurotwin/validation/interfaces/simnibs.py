"""Normal projection behind explicit physical-coordinate and registration checks."""

import numpy as np

from ..structural.observation import registration_gate


def project_normal_gain(cartesian_gain, normals, *, frame, channel_names, expected_channels):
    """Pure projection matching the frozen interface convention, not an FEM solve."""
    L = np.asarray(cartesian_gain, float)
    n = np.asarray(normals, float)
    if frame != "native_scanner_RAS_mm":
        raise ValueError("Verified native scanner-RAS frame required")
    if list(channel_names) != list(expected_channels) or len(channel_names) != len(
        set(channel_names)
    ):
        raise ValueError("Exact unique channel order required")
    if (
        L.ndim != 3
        or L.shape[2] != 3
        or n.shape != (L.shape[1], 3)
        or L.shape[0] != len(channel_names)
    ):
        raise ValueError("Cartesian gain must have channel,vertex,XYZ layout")
    if not np.isfinite(L).all() or not np.isfinite(n).all():
        raise ValueError("Finite forward data required")
    length = np.linalg.norm(n, axis=1)
    if np.any(length <= 0):
        raise ValueError("Zero source normal")
    return np.einsum("cvi,vi->cv", L, n / length[:, None])


def load_registered_forward(path, *, sha256, registration):
    """Optional read-only interface; actual-runtime use needs independent approval."""
    from ..artifacts import sha

    registration_gate(registration, sha256)
    if sha(path) != sha256:
        raise ValueError("Forward input hash mismatch")
    import importlib.metadata as md

    if md.version("simnibs") != "4.6.0":
        raise RuntimeError("Unreviewed SimNIBS version")
    from simnibs.eeg.forward import prepare_forward

    return prepare_forward(str(path), apply_average_proj=True)
