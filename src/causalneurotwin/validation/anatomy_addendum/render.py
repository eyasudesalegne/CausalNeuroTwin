"""Focused native-RAS evidence; never grants anatomical acceptance."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from causalneurotwin.validation.anatomy.geometry import plane_segments
from causalneurotwin.validation.anatomy.render import normalized

ENDS = [("P", "A", "I", "S"), ("L", "R", "I", "S"), ("L", "R", "P", "A")]


def save(fig, path):
    if path.exists():
        raise FileExistsError(path)
    fig.text(
        0.5,
        0.01,
        "Header-derived sides. Unaccepted anatomical evidence; independent review pending.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cap(ids, n):
    return ids[np.linspace(0, len(ids) - 1, n, dtype=int)] if len(ids) > n else ids


def background(ax, axis, center, original, tissues, surfaces, options):
    other = [j for j in range(3) if j != axis]
    half, step = options["halfwidth_mm"], options["step_mm"]
    u = np.arange(center[other[0]] - half, center[other[0]] + half + step / 2, step)
    v = np.arange(center[other[1]] - half, center[other[1]] + half + step / 2, step)
    uu, vv = np.meshgrid(u, v)
    grid = np.empty(uu.shape + (3,))
    grid[..., axis] = center[axis]
    grid[..., other[0]], grid[..., other[1]] = uu, vv
    extent = [u[0] - step / 2, u[-1] + step / 2, v[0] - step / 2, v[-1] + step / 2]
    ax.imshow(
        normalized(original.sample(grid)),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    labels = tissues.sample(grid, order=0, outside=0)
    for tag, color in ((1, "orange"), (2, "lime"), (3, "dodgerblue"), (7, "yellow"), (8, "pink")):
        mask = labels == tag
        if mask.any() and not mask.all():
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors=[color],
                linewidths=0.5,
            )
    for hemi, (vertices, faces) in surfaces.items():
        segments = plane_segments(vertices, faces, axis, center[axis])
        if len(segments):
            ax.add_collection(
                LineCollection(
                    segments[:, :, other],
                    colors="cyan" if hemi == "lh" else "magenta",
                    linewidths=0.4,
                )
            )
    left, right, bottom, top = ENDS[axis]
    ax.set(
        xlim=extent[:2],
        ylim=extent[2:],
        xlabel=f"{left} to {right} (RAS mm)",
        ylabel=f"{bottom} to {top} (RAS mm)",
        title=f"{['Sagittal', 'Coronal', 'Axial'][axis]} {center[axis]:.2f} mm",
    )
    ax.set_aspect("equal")
    ax.plot(center[other[0]], center[other[1]], "+", color="white")
    return other


def triplanar(path, title, center, original, tissues, surfaces, options, groups=()):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, ax in enumerate(axes):
        other = background(ax, axis, np.asarray(center), original, tissues, surfaces, options)
        for label, points, color in groups:
            points = np.asarray(points).reshape(-1, 3)
            ids = np.flatnonzero(
                abs(points[:, axis] - center[axis]) <= options["slab_halfwidth_mm"]
            )
            ids = cap(ids, options["scatter_cap"])
            ax.scatter(points[ids, other[0]], points[ids, other[1]], s=4, c=color, label=label)
    if groups:
        axes[0].legend(fontsize=6)
    fig.suptitle(
        title
        + "\nWM orange; GM green; CSF blue; compact bone yellow; surfaces LH cyan / RH magenta"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    save(fig, path)


def parcel_figure(path, row, vertices, surfaces, masks, distance, original, tissues, options):
    seeds, rejected, added = masks
    center = np.median(vertices[seeds], axis=0)
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.05],
        hspace=0.35,
        wspace=0.3,
        left=0.07,
        right=0.94,
        bottom=0.14,
        top=0.87,
    )
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)])
    color_axis = fig.add_subplot(grid[:, 3])
    image = None
    for axis in range(3):
        other = background(axes[0, axis], axis, center, original, tissues, surfaces, options)
        # All-point lower projections expose historical opposite-side samples even off slice.
        for panel in (0, 1):
            ax = axes[panel, axis]
            near = (
                abs(vertices[:, axis] - center[axis]) <= options["slab_halfwidth_mm"]
                if panel == 0
                else np.ones(len(vertices), bool)
            )
            for mask, color, label in (
                (seeds, "cyan", "Stored correct-side direct seeds"),
                (rejected, "magenta", "Historical rejected opposite-side samples"),
            ):
                ids = cap(np.flatnonzero(mask & near), options["scatter_cap"])
                ax.scatter(
                    vertices[ids, other[0]], vertices[ids, other[1]], s=4, c=color, label=label
                )
            ids = cap(np.flatnonzero(added & near), options["scatter_cap"])
            image = ax.scatter(
                vertices[ids, other[0]],
                vertices[ids, other[1]],
                s=4,
                c=distance[ids],
                cmap="autumn_r",
                vmin=0,
                vmax=max(float(distance[added].max()), 1) if added.any() else 1,
                label="Unaccepted geodesic additions",
            )
            if panel == 1:
                ax.set(
                    xlabel=f"RAS {'xyz'[other[0]]} mm",
                    ylabel=f"RAS {'xyz'[other[1]]} mm",
                    title="All-point projection\n(not an anatomical slice)",
                )
                points = vertices[seeds | rejected | added][:, other]
                mid = (points.min(0) + points.max(0)) / 2
                half = max(float(np.ptp(points, axis=0).max() / 2) * 1.1, 5)
                ax.set(xlim=(mid[0] - half, mid[0] + half), ylim=(mid[1] - half, mid[1] + half))
                ax.set_aspect("equal")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, fontsize=8)
    if image is not None:
        fig.colorbar(image, cax=color_axis, label="Stored distance from trusted seeds (mm)")
    fig.suptitle(
        f"{row['atlas_id']} {row['label']}: seeds and unaccepted historical completion\n"
        f"Trusted fraction {float(row['trusted_fraction']):.3f}; expansion {float(row['expansion_ratio']):.3f}; "
        "lower panels show capped projections, metrics use all vertices"
    )
    save(fig, path)


def mesh_wireframe(path, eid, vertices, tag):
    xyz = np.asarray(vertices)
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(141, projection="3d")
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ax.plot(*xyz[[i, j]].T, color="black")
    center = xyz.mean(0)
    half = max(np.ptp(xyz, axis=0).max() / 2, 1e-6)
    ax.set(
        xlim=(center[0] - half, center[0] + half),
        ylim=(center[1] - half, center[1] + half),
        zlim=(center[2] - half, center[2] + half),
        xlabel="RAS x mm",
        ylabel="RAS y mm",
        zlabel="RAS z mm",
    )
    ax.set_box_aspect((1, 1, 1))
    for n, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
        a = fig.add_subplot(1, 4, n)
        for u, v in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            a.plot(xyz[[u, v], i], xyz[[u, v], j], "-o", ms=3, color="black")
        a.set(xlabel=f"RAS {'xyz'[i]} mm", ylabel=f"RAS {'xyz'[j]} mm")
        a.set_aspect("equal", adjustable="datalim")
        for k in range(4):
            a.annotate(str(k), xyz[k, [i, j]])
    fig.suptitle(
        f"Element {eid}, tissue {tag}: actual edges; equal physical scale, no thickness exaggeration"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    save(fig, path)


def surface_distributions(path, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hemi, arrays in results.items():
        mask = arrays["disagreement"] & np.isfinite(arrays["boundary_distance_mm"])
        d = arrays["boundary_distance_mm"][mask]
        if len(d):
            axes[0].hist(d, bins=40, histtype="step", label=hemi)
            axes[1].hist(
                d, bins=40, weights=arrays["vertex_area_mm2"][mask], histtype="step", label=hemi
            )
    for ax in axes:
        ax.set_xlabel("Exact distance to labeled-GM voxel-cell boundary (mm)")
        ax.legend()
    axes[0].set_ylabel("Discordant vertices")
    axes[1].set_ylabel("Barycentric surface area (mm2)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


def mesh_overview(path, rows, options):
    """Projections expose warning locations; no points are claimed to lie in a slice."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for tag in sorted({r["tissue_tag"] for r in rows}):
        xyz = np.array(
            [[r[k] for k in ("x_mm", "y_mm", "z_mm")] for r in rows if r["tissue_tag"] == tag]
        )
        ids = cap(np.arange(len(xyz)), options["scatter_cap"])
        for ax, (i, j) in zip(axes, ((1, 2), (0, 2), (0, 1))):
            ax.scatter(xyz[ids, i], xyz[ids, j], s=2, label=f"Tissue {tag}")
            ax.set(xlabel=f"RAS {'xyz'[i]} mm", ylabel=f"RAS {'xyz'[j]} mm")
            ax.set_aspect("equal")
    axes[0].legend(fontsize=7, markerscale=3)
    fig.suptitle(
        "Warning-element centroids: all-point projections, not anatomical slices\n"
        "Rendering capped per tissue; exact counts and all centroids are saved"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    save(fig, path)
