"""Native physical-coordinate evidence; matplotlib is only imported on compute or synthetic tests."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from .geometry import apply, corners, plane_segments
from .io import PENDING, write_csv, write_json

AXIS_NAMES = ["Sagittal", "Coronal", "Axial"]
ENDS = {0: ("P", "A", "I", "S"), 1: ("L", "R", "I", "S"), 2: ("L", "R", "P", "A")}
TISSUES = [
    (1, "WM", "#ffbf00"),
    (2, "GM", "#ff5555"),
    (3, "CSF", "#5599ff"),
    (4, "bone", "#bfff00"),
    (7, "compact bone", "#dddd33"),
    (8, "spongy bone", "#ee88ff"),
]


def normalized(data):
    good = data[np.isfinite(data) & (data > 0)]
    lo, hi = np.quantile(good, [0.02, 0.98]) if len(good) > 1 else (0, 1)
    return np.clip((np.nan_to_num(data) - lo) / max(float(hi - lo), 1e-6), 0, 1)


def plane_grid(original, axis, coordinate, step):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Display sampling step must be positive and finite")
    world = apply(original.affine, corners(original.shape))
    other = [i for i in range(3) if i != axis]
    u = np.arange(world[:, other[0]].min(), world[:, other[0]].max() + step / 2, step)
    v = np.arange(world[:, other[1]].min(), world[:, other[1]].max() + step / 2, step)
    uu, vv = np.meshgrid(u, v)
    grid = np.empty(uu.shape + (3,))
    grid[..., axis] = coordinate
    grid[..., other[0]] = uu
    grid[..., other[1]] = vv
    return grid, [u[0] - step / 2, u[-1] + step / 2, v[0] - step / 2, v[-1] + step / 2], other


def _decorate(ax, axis):
    left, right, bottom, top = ENDS[axis]
    ax.set_xlabel(f"{left} to {right} (RAS world mm)")
    ax.set_ylabel(f"{bottom} to {top} (RAS world mm)")
    ax.set_aspect("equal")


def _contour(ax, mask, extent, color, width=0.55):
    if mask.any() and not mask.all():
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors=[color],
            linewidths=width,
        )


def save_figure(fig, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    fig.text(
        0.5,
        0.02,
        "Header-derived orientation. Computational visualization; independent anatomical assessment pending.",
        ha="center",
        fontsize=7,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": "CON020 anatomy v1.0.0"})
    plt.close(fig)


def anatomical_figure(volumes, surfaces, pull, axis, coordinate, path, step=1.5, highlight=None):
    raw = volumes["original_t1"]
    grid, extent, other = plane_grid(raw, axis, coordinate, step)
    background = normalized(raw.sample(grid))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax in axes:
        ax.imshow(background, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
        _decorate(ax, axis)
    axes[0].set_title("Original subject T1")
    labels = volumes["simnibs_tissues"].sample(grid, order=0, outside=0)
    for label, _name, color in TISSUES:
        _contour(axes[1], labels == label, extent, color)
    for hemi, (vertices, faces) in surfaces.items():
        segments = plane_segments(vertices, faces, axis, coordinate)
        if len(segments):
            axes[1].add_collection(
                LineCollection(
                    segments[:, :, other],
                    colors="cyan" if hemi == "lh" else "magenta",
                    linewidths=0.45,
                )
            )
    axes[1].set_title("CHARM boundaries + central surfaces")
    handles = [Line2D([0], [0], color=c, label=n) for _, n, c in TISSUES]
    handles += [
        Line2D([0], [0], color="cyan", label="LH central"),
        Line2D([0], [0], color="magenta", label="RH central"),
    ]
    axes[1].legend(handles=handles, fontsize=6, loc="upper right")
    acpc_grid = apply(np.linalg.inv(pull), grid)
    qseg = volumes["qsiprep_aseg"].sample(acpc_grid, order=0, outside=0)
    # All categorical transitions are shown; no unverified QSIPrep tissue LUT is assumed.
    boundary = np.zeros(qseg.shape, dtype=bool)
    boundary[1:, :] |= qseg[1:, :] != qseg[:-1, :]
    boundary[:, 1:] |= qseg[:, 1:] != qseg[:, :-1]
    overlay = np.zeros(boundary.shape + (4,))
    overlay[boundary] = [0.3, 1, 0.3, 0.8]
    axes[2].imshow(overlay, origin="lower", extent=extent)
    mask = volumes["qsiprep_mask"].sample(acpc_grid, order=0, outside=0)
    _contour(axes[2], mask > 0, extent, "orange", 0.9)
    axes[2].set_title("QSIPrep aseg transitions / mask (orange)")
    registered = normalized(volumes["qsiprep_t1"].sample(acpc_grid))
    ii, jj = np.indices(background.shape)
    checker = np.where((ii // 16 + jj // 16) % 2, background, registered)
    axes[3].imshow(checker, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    axes[3].set_title("Original / ACPC T1 checkerboard\nsame-source derivative consistency only")
    if highlight is not None:
        for ax in axes:
            ax.plot(highlight[other[0]], highlight[other[1]], "+", color="yellow", ms=8)
    fig.suptitle(f"{AXIS_NAMES[axis]} at native RAS {['x', 'y', 'z'][axis]}={coordinate:.2f} mm")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    save_figure(fig, path)


def surface_candidates(volumes, surfaces, output, max_rows=500):
    all_rows, counts = [], {}
    for hemi, (vertices, _) in surfaces.items():
        tissue = volumes["simnibs_tissues"].sample(vertices, order=0)
        mismatch = ~np.isfinite(tissue) | (tissue != 2)
        counts[hemi] = {
            "vertices": len(vertices),
            "central_vertices_not_in_gm": int(mismatch.sum()),
            "fraction": float(mismatch.mean()),
            "outside_tissue_fov": int((~np.isfinite(tissue)).sum()),
        }
        indices = np.flatnonzero(mismatch)
        indices = (
            indices[np.linspace(0, len(indices) - 1, min(max_rows // 2, len(indices)), dtype=int)]
            if len(indices)
            else indices
        )
        for i in indices:
            all_rows.append(
                {
                    "hemisphere": hemi,
                    "vertex_index": int(i),
                    "x_mm": float(vertices[i, 0]),
                    "y_mm": float(vertices[i, 1]),
                    "z_mm": float(vertices[i, 2]),
                    "sampled_charm_tissue": float(tissue[i]) if np.isfinite(tissue[i]) else "",
                    "reason": "central_surface_tissue_discordance_candidate",
                    "evidence_category": PENDING,
                    "assessment_status": "pending",
                }
            )
    write_csv(
        output / "boundary_candidates.csv",
        all_rows,
        [
            "hemisphere",
            "vertex_index",
            "x_mm",
            "y_mm",
            "z_mm",
            "sampled_charm_tissue",
            "reason",
            "evidence_category",
            "assessment_status",
        ],
    )
    write_json(
        output / "boundary_candidates.json",
        {
            "evidence_category": PENDING,
            "assessment_status": "pending",
            "by_hemisphere": counts,
            "csv_sampling": "Deterministic spread in vertex index; CSV capped, counts use all vertices",
            "limitation": "Central surface/tissue disagreement is a review candidate, not proven anatomical error.",
        },
    )
    return all_rows


def parcel_figures(
    volumes, surfaces, pull, parcels, direct_path, repaired_path, output, limit=12, step=1.5
):
    vertices = np.concatenate([surfaces["lh"][0], surfaces["rh"][0]])
    direct = np.load(direct_path, allow_pickle=False)
    repaired = np.load(repaired_path, allow_pickle=False)
    if direct.shape != (len(vertices),) or repaired.shape != direct.shape:
        raise ValueError("Stored surface labels do not match the verified surface vertex layout")
    nleft = len(surfaces["lh"][0])
    for row in [r for r in parcels if r["triage_reasons"]][:limit]:
        aid = row["atlas_id"]
        correct_hemi = (
            np.arange(len(vertices)) < nleft
            if row["label"].endswith("_L")
            else np.arange(len(vertices)) >= nleft
        )
        seeds = (direct == aid) & correct_hemi
        if not seeds.any():
            continue
        center = np.median(vertices[seeds], axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for axis, ax in enumerate(axes):
            grid, extent, other = plane_grid(volumes["original_t1"], axis, center[axis], step)
            ax.imshow(
                normalized(volumes["original_t1"].sample(grid)),
                cmap="gray",
                origin="lower",
                extent=extent,
            )
            native_labels = volumes["brainnetome_native"].sample(grid, order=0, outside=0)
            _contour(ax, native_labels == aid, extent, "yellow", 1)
            near = abs(vertices[:, axis] - center[axis]) <= step / 2
            for mask, color, label in [
                (near & seeds, "cyan", "Stored direct correct-side samples"),
                (
                    near & (repaired == aid) & (direct != repaired),
                    "red",
                    "Stored geodesic additions",
                ),
                (near & (direct == aid) & ~correct_hemi, "magenta", "Stored opposite-side samples"),
            ]:
                ids = np.flatnonzero(mask)
                if len(ids) > 1500:
                    ids = ids[np.linspace(0, len(ids) - 1, 1500, dtype=int)]
                ax.scatter(
                    vertices[ids, other[0]], vertices[ids, other[1]], s=2, c=color, label=label
                )
            _decorate(ax, axis)
            ax.set_title(f"{AXIS_NAMES[axis]} {center[axis]:.2f} mm")
        axes[0].legend(fontsize=6)
        fig.suptitle(
            f"{aid} {row['label']}: {row['triage_reasons']}\n"
            f"trusted fraction={row['trusted_fraction']:.3f}; expansion={row['expansion_ratio']:.3f}"
        )
        fig.tight_layout(rect=(0, 0.13, 1, 0.90))
        rel = f"figures/parcel_{aid}.png"
        save_figure(fig, output / rel)
        row["evidence_image"] = rel


def mesh_histogram(values, output):
    fig, axes = plt.subplots(1, len(values), figsize=(18, 4))
    for ax, (key, a) in zip(axes, values.items()):
        good = a[np.isfinite(a)]
        if len(good):
            # Plot central distribution; full extremes/percentiles remain in JSON.
            lo, hi = np.quantile(good, [0, 0.999])
            ax.hist(good, bins=60, range=(lo, hi) if hi > lo else None, log=True)
        ax.set_title(key)
        ax.set_ylabel("element count (log)")
        ax.set_xlabel("Through p99.9; full extremes in JSON")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save_figure(fig, output / "figures/mesh_shape_distributions.png")
