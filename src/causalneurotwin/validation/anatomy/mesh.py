"""Bounded-memory Gmsh 2.2 linear-tetra shape quality; no remeshing."""

import heapq
import struct

import numpy as np

from .io import CONSISTENCY, PENDING, write_csv, write_json


def _line(handle):
    line = handle.readline()
    if not line:
        raise ValueError("Unexpected end of Gmsh file")
    return line.strip()


def _read_exact(handle, n):
    data = handle.read(n)
    if len(data) != n:
        raise ValueError("Truncated Gmsh binary block")
    return data


class Gmsh22:
    """Only the verified Gmsh 2.2 triangle/tetra format is supported; reject other types."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.h = open(self.path, "rb")
        try:
            if _line(self.h) != b"$MeshFormat":
                raise ValueError("Missing Gmsh format header")
            version, binary, size = _line(self.h).split()
            if version != b"2.2" or size != b"8" or binary not in (b"0", b"1"):
                raise ValueError("Require Gmsh 2.2 with 8-byte reals")
            self.binary = binary == b"1"
            self.endian = "<"
            if self.binary:
                marker = _read_exact(self.h, 4)
                self.endian = "<" if struct.unpack("<i", marker)[0] == 1 else ">"
                if struct.unpack(self.endian + "i", marker)[0] != 1:
                    raise ValueError("Invalid endianness marker")
            end = _line(self.h)
            if end == b"":
                end = _line(self.h)
            if end != b"$EndMeshFormat" or _line(self.h) != b"$Nodes":
                raise ValueError("Unsupported Gmsh sections before nodes")
            n = int(_line(self.h))
            if n <= 0:
                raise ValueError("Empty node array")
            if self.binary:
                dtype = np.dtype([("id", self.endian + "i4"), ("xyz", self.endian + "f8", (3,))])
                a = np.frombuffer(_read_exact(self.h, n * dtype.itemsize), dtype=dtype)
                self.ids = a["id"].astype(np.int64)
                self.xyz = a["xyz"].astype(float)
                end = _line(self.h)
                if end == b"":
                    end = _line(self.h)
            else:
                a = np.array([list(map(float, _line(self.h).split())) for _ in range(n)])
                self.ids, self.xyz = a[:, 0].astype(np.int64), a[:, 1:]
                end = _line(self.h)
            if end != b"$EndNodes" or _line(self.h) != b"$Elements":
                raise ValueError("Malformed node/element section")
            if not np.isfinite(self.xyz).all() or self.xyz.shape != (n, 3):
                raise ValueError("Invalid node coordinates")
            self.order = np.argsort(self.ids)
            self.sorted_ids = self.ids[self.order]
            if np.any(np.diff(self.sorted_ids) == 0):
                raise ValueError("Duplicate node IDs")
            self.n_elements = int(_line(self.h))
            return self
        except Exception:
            self.h.close()
            raise

    def __exit__(self, *args):
        self.h.close()

    def _indices(self, ids):
        pos = np.searchsorted(self.sorted_ids, ids)
        if np.any(pos >= len(self.ids)) or not np.array_equal(self.sorted_ids[pos], ids):
            raise ValueError("Element references unknown node IDs")
        return self.order[pos]

    def tetra_chunks(self, chunk_size=50000):
        seen = 0
        if self.binary:
            while seen < self.n_elements:
                kind, count, tags = struct.unpack(self.endian + "3i", _read_exact(self.h, 12))
                if kind not in (2, 4) or count <= 0 or tags < 0 or seen + count > self.n_elements:
                    raise ValueError(f"Unsupported/malformed element block type {kind}")
                width = 1 + tags + ({2: 3, 4: 4}[kind])
                for start in range(0, count, chunk_size):
                    n = min(chunk_size, count - start)
                    a = np.frombuffer(
                        _read_exact(self.h, n * width * 4), dtype=self.endian + "i4"
                    ).reshape(n, width)
                    if kind == 4:
                        tag = a[:, 1] if tags else np.zeros(n, dtype=int)
                        yield a[:, 0].copy(), tag.copy(), self.xyz[self._indices(a[:, 1 + tags :])]
                seen += count
            end = _line(self.h)
            if end == b"":
                end = _line(self.h)
        else:
            ids, tags_out, vertices = [], [], []
            for _ in range(self.n_elements):
                a = list(map(int, _line(self.h).split()))
                if len(a) < 3:
                    raise ValueError("Malformed ASCII element")
                kind, tags = a[1], a[2]
                if kind not in (2, 4) or len(a) != 3 + tags + {2: 3, 4: 4}[kind]:
                    raise ValueError("Unsupported ASCII element")
                if kind == 4:
                    ids.append(a[0])
                    tags_out.append(a[3] if tags else 0)
                    vertices.append(a[3 + tags :])
                    if len(ids) == chunk_size:
                        yield (
                            np.array(ids),
                            np.array(tags_out),
                            self.xyz[self._indices(np.array(vertices))],
                        )
                        ids, tags_out, vertices = [], [], []
            if ids:
                yield np.array(ids), np.array(tags_out), self.xyz[self._indices(np.array(vertices))]
            end = _line(self.h)
        if end != b"$EndElements":
            raise ValueError("Malformed end of element section")


def tetra_quality(vertices):
    """Mean ratio, minimum scaled corner Jacobian, edge ratio and internal dihedrals.

    Mean ratio = 12*(3*abs(V))**(2/3)/sum(six squared edge lengths).
    Scaled Jacobian = sqrt(2)*min_corner(det(J)/product(three corner edge lengths)).
    Ideal regular tetra: mean ratio=1, scaled Jacobian=1, edge ratio=1,
    internal dihedral=acos(1/3)=70.5288 degrees. Inversion remains signed in SJ.
    """
    t = np.asarray(vertices, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (4, 3):
        raise ValueError("Expected (N,4,3) tetrahedra")
    edges = np.stack(
        [t[:, j] - t[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], 1
    )
    lengths = np.linalg.norm(edges, axis=2)
    det = np.linalg.det(np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]], 2))
    products = []
    for i in range(4):
        products.append(
            np.prod([np.linalg.norm(t[:, j] - t[:, i], axis=1) for j in range(4) if j != i], axis=0)
        )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mr = 12 * np.power(np.abs(det) / 2, 2 / 3) / np.sum(lengths**2, axis=1)
        sj = np.min(np.sqrt(2) * det[:, None] / np.stack(products, 1), axis=1)
        aspect = lengths.max(1) / lengths.min(1)
        normals = []
        for a, b, c, opposite in ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3)):
            n = np.cross(t[:, b] - t[:, a], t[:, c] - t[:, a])
            dot = np.einsum("ij,ij->i", n, t[:, opposite] - t[:, a])
            n *= np.where(dot > 0, -1, 1)[:, None]
            normals.append(n / np.linalg.norm(n, axis=1)[:, None])
        angles = np.stack(
            [
                np.degrees(
                    np.pi - np.arccos(np.clip(np.einsum("ij,ij->i", normals[i], normals[j]), -1, 1))
                )
                for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ],
            1,
        )
    return {
        "mean_ratio": mr,
        "scaled_jacobian": sj,
        "edge_ratio": aspect,
        "min_dihedral_deg": angles.min(1),
        "max_dihedral_deg": angles.max(1),
    }


def distribution(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    out = {"n": len(a), "nonfinite": int(len(a) - len(finite))}
    if len(finite):
        out.update(
            dict(
                zip(
                    ["min", "p01", "p05", "p50", "p95", "p99", "max"],
                    np.quantile(finite, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).tolist(),
                )
            )
        )
    return out


def mesh_report(path, output, thresholds):
    values, tags, worst = {}, [], []
    serial = 0
    with Gmsh22(path) as mesh:
        for ids, tag, xyz in mesh.tetra_chunks():
            q = tetra_quality(xyz)
            tags.append(tag)
            for k, v in q.items():
                values.setdefault(k, []).append(v)
            score = np.nan_to_num(q["mean_ratio"], nan=-1)
            take = np.argsort(score)[: min(100, len(score))]
            for i in take:
                row = {
                    "element_id": int(ids[i]),
                    "tissue_tag": int(tag[i]),
                    "x_mm": float(xyz[i, :, 0].mean()),
                    "y_mm": float(xyz[i, :, 1].mean()),
                    "z_mm": float(xyz[i, :, 2].mean()),
                    **{k: float(v[i]) for k, v in q.items()},
                    "evidence_category": PENDING,
                    "assessment_status": "pending",
                }
                heapq.heappush(worst, (-float(score[i]), serial, row))
                serial += 1
                if len(worst) > 100:
                    heapq.heappop(worst)
    if not tags:
        raise ValueError("No linear tetrahedra in mesh")
    tag = np.concatenate(tags)
    values = {k: np.concatenate(v) for k, v in values.items()}
    rows = [v[2] for v in sorted(worst, key=lambda v: -v[0])]
    fields = [
        "element_id",
        "tissue_tag",
        "x_mm",
        "y_mm",
        "z_mm",
        *values,
        "evidence_category",
        "assessment_status",
    ]
    write_csv(output / "mesh_worst_elements.csv", rows, fields)
    report = {
        "evidence_category": CONSISTENCY,
        "assessment_status": "verified",
        "scope": "New shape distributions computed; anatomical/solver acceptance pending",
        "independent_anatomical_assessment": "pending",
        "tetrahedra_evaluated": len(tag),
        "global": {k: distribution(v) for k, v in values.items()},
        "by_tissue": {
            str(t): {k: distribution(v[tag == t]) for k, v in values.items()}
            for t in np.unique(tag)
        },
        "triage_thresholds_provisional": thresholds,
        "triage_counts": {
            "mean_ratio_below": int(np.sum(values["mean_ratio"] < thresholds["mean_ratio_below"])),
            "scaled_jacobian_below": int(
                np.sum(values["scaled_jacobian"] < thresholds["scaled_jacobian_below"])
            ),
            "min_dihedral_below_deg": int(
                np.sum(values["min_dihedral_deg"] < thresholds["min_dihedral_below_deg"])
            ),
            "max_dihedral_above_deg": int(
                np.sum(values["max_dihedral_deg"] > thresholds["max_dihedral_above_deg"])
            ),
        },
        "limitation": "Shape alone does not validate anatomical labels, FEM convergence or biological accuracy.",
    }
    if any(v["nonfinite"] for v in report["global"].values()):
        report["evidence_category"] = PENDING
        report["assessment_status"] = "indeterminate"
    write_json(output / "mesh_quality.json", report)
    return report, values, rows
