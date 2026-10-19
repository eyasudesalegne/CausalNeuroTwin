"""Explicit NIfTI geometry and ITK physical-point transforms (RAS millimeters)."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def apply(matrix, points):
    p = np.asarray(points, dtype=float)
    return p @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def validate_affine(matrix):
    a = np.asarray(matrix, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Affine must be finite 4x4")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError("Invalid homogeneous affine")
    if abs(np.linalg.det(a[:3, :3])) < 1e-12:
        raise ValueError("Singular spatial affine")
    return a


def corners(shape):
    return np.asarray(list(product(*[(0, n - 1) for n in shape[:3]])), dtype=float)


@dataclass
class Volume:
    image: object
    affine: np.ndarray
    key: str
    space: str
    report: dict
    _data: object = None

    @property
    def shape(self):
        return self.image.shape[:3]

    @property
    def data(self):
        if self._data is None:
            data = np.asanyarray(self.image.dataobj)
            if data.ndim == 4:
                data = data[..., 0]
            self._data = np.asarray(data, dtype=np.float32)
        return self._data

    def sample(self, ras_points, order=1, outside=np.nan):
        ijk = apply(np.linalg.inv(self.affine), ras_points)
        shape = ijk.shape
        value = map_coordinates(
            self.data,
            ijk.reshape(-1, 3).T,
            order=order,
            mode="constant",
            cval=outside,
            prefilter=False,
        )
        return value.reshape(shape[:-1])

    def contains(self, point):
        v = apply(np.linalg.inv(self.affine), np.asarray(point))
        return bool(np.all(v >= 0) and np.all(v <= np.asarray(self.shape) - 1))


def load_volume(key, entry, tolerance_mm=0.001):
    im = nib.load(str(entry["path"]))
    if len(im.shape) != 3 and not (len(im.shape) == 4 and im.shape[3] == 1):
        raise ValueError(f"{key}: require 3D or singleton 4D NIfTI, got {im.shape}")
    if im.header.get_xyzt_units()[0] != "mm":
        raise ValueError(f"{key}: spatial units must explicitly be mm")
    q, qc = im.get_qform(coded=True)
    s, sc = im.get_sform(coded=True)
    selection = entry["form"]
    if selection not in ("sform", "qform"):
        raise ValueError("Explicit qform/sform selection required; no fallback")
    affine, code = (s, sc) if selection == "sform" else (q, qc)
    if int(code) == 0 or affine is None:
        raise ValueError(f"{key}: selected {selection} is uncoded")
    affine = validate_affine(affine)
    discrepancy = None
    if int(qc) and int(sc):
        discrepancy = float(
            np.linalg.norm(apply(q, corners(im.shape)) - apply(s, corners(im.shape)), axis=1).max()
        )
    issues = []
    if discrepancy is not None and discrepancy > tolerance_mm:
        issues.append("qform_sform_disagreement_requires_review")
    expected = entry.get("expected_header", {})
    if expected.get("shape") and list(im.shape) != expected["shape"]:
        issues.append("shape_changed_since_read_only_inventory")
    if expected.get("axcodes") and list(nib.aff2axcodes(affine)) != expected["axcodes"]:
        issues.append("orientation_changed_since_read_only_inventory")
    for name, value in [("qform_code", int(qc)), ("sform_code", int(sc))]:
        if name in expected and value != expected[name]:
            issues.append(name + "_changed_since_inventory")
    if expected.get("selected_affine") is not None:
        distance = np.linalg.norm(
            apply(affine, corners(im.shape))
            - apply(expected["selected_affine"], corners(im.shape)),
            axis=1,
        ).max()
        if distance > tolerance_mm:
            issues.append("selected_affine_changed_since_inventory")
    basis = affine[:3, :3]
    scales = np.linalg.norm(basis, axis=0)
    report = {
        "evidence_category": (
            "validation_pending_failed_or_indeterminate"
            if issues
            else "computational_consistency_verified"
        ),
        "assessment_status": "indeterminate" if issues else "verified",
        "independent_anatomical_assessment": "pending",
        "path": str(Path(entry["path"]).resolve()),
        "shape": list(im.shape),
        "dtype": str(im.get_data_dtype()),
        "space": entry["space"],
        "units": "mm",
        "selected_form": selection,
        "qform_code": int(qc),
        "sform_code": int(sc),
        "affine_ras_mm": affine.tolist(),
        "axcodes": list(nib.aff2axcodes(affine)),
        "determinant": float(np.linalg.det(basis)),
        "voxel_axis_lengths_mm": scales.tolist(),
        "axis_cosines": ((basis / scales).T @ (basis / scales)).tolist(),
        "qform_sform_max_corner_difference_mm": discrepancy,
        "world_bounds_mm": [
            apply(affine, corners(im.shape)).min(0).tolist(),
            apply(affine, corners(im.shape)).max(0).tolist(),
        ],
        "issues": issues,
        "interpretation": "Header geometry only; anatomical left/right identity is not certified.",
    }
    return Volume(im, affine, key, entry["space"], report)


def itk_matrix(path):
    """Read one ITK MATLAB-v4 Euler3D or Affine transform. Returned point map is RAS->RAS.

    ANTs image-transform filenames describe image motion, NOT physical-point direction.
    The configured forward image pull map maps ACPC output points to native input points.
    """
    d = loadmat(str(path))
    names = [k for k in d if not k.startswith("__") and k != "fixed"]
    if len(names) != 1 or "fixed" not in d:
        raise ValueError("Require exactly one ITK transform with fixed parameters")
    name = names[0]
    p = np.asarray(d[name], dtype=float).ravel()
    fixed = np.asarray(d["fixed"], dtype=float).ravel()
    if not np.isfinite(p).all() or not np.isfinite(fixed).all() or len(fixed) < 3:
        raise ValueError("Invalid ITK parameters")
    c = fixed[:3]
    if name.startswith("Euler3DTransform") and len(p) == 6:
        if len(fixed) != 4 or fixed[3] not in (0, 1):
            raise ValueError("Euler3D requires explicit ComputeZYX fixed parameter")
        x, y, z = p[:3]
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        rotation = rz @ (ry @ rx if fixed[3] else rx @ ry)
        translation = p[3:]
    elif name.startswith("AffineTransform") and len(p) == 12:
        rotation, translation = p[:9].reshape(3, 3), p[9:]
    else:
        raise ValueError(f"Unsupported ITK transform: {name}")
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = c + translation - rotation @ c
    return validate_affine(LPS_RAS @ m @ LPS_RAS)


def point_map(source_space, target_space, acpc_to_native):
    if source_space not in ("native", "acpc") or target_space not in ("native", "acpc"):
        raise ValueError("Unknown physical space")
    if source_space == target_space:
        return np.eye(4)
    return acpc_to_native if source_space == "acpc" else np.linalg.inv(acpc_to_native)


def load_surface(entry):
    im = nib.load(str(entry["path"]))
    arrays = [a for a in im.darrays if int(a.intent) == 1008]
    faces = [a for a in im.darrays if int(a.intent) == 1009]
    if len(arrays) != 1 or len(faces) != 1:
        raise ValueError("Require one GIFTI pointset and one triangle array")
    a = arrays[0]
    if a.coordsys is None or int(a.coordsys.xformspace) != 1:
        raise ValueError("Surface must explicitly transform to scanner-anatomical RAS")
    vertices = apply(validate_affine(a.coordsys.xform), a.data)
    triangles = np.asarray(faces[0].data, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid surface vertices")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
    ):
        raise ValueError("Invalid triangle indices")
    return vertices, triangles


def plane_segments(vertices, faces, axis, coordinate):
    """True triangle/plane intersections; no projection of a thick vertex slab."""
    d = vertices[:, axis] - coordinate
    fd = d[faces]
    tri = vertices[faces[(fd.min(1) <= 0) & (fd.max(1) > 0)]]
    result = []
    for t in tri:
        delta = t[:, axis] - coordinate
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (delta[i] <= 0 < delta[j]) or (delta[j] <= 0 < delta[i]):
                points.append(t[i] + (-delta[i] / (delta[j] - delta[i])) * (t[j] - t[i]))
        if len(points) == 2 and not np.allclose(points[0], points[1]):
            result.append(points)
    return np.asarray(result, dtype=float).reshape(-1, 2, 3)
