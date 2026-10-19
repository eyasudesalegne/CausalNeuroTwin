"""New native-RAS geometry evidence; no illustrative MRI or anatomical acceptance."""

from pathlib import Path

import numpy as np


def mask_figure(path, points, masks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    masks = dict(masks)
    masks["reviewed_excluded"] = masks["original_trusted"] & ~masks["reviewed_conservative"]
    points = np.asarray(points)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i, j, name) in zip(axes, [(1, 2, "Sagittal"), (0, 2, "Coronal"), (0, 1, "Axial")]):
        for key, color in [
            ("original_trusted", "silver"),
            ("unresolved_component", "darkorange"),
            ("sensitivity_core", "navy"),
            ("reviewed_excluded", "crimson"),
        ]:
            ids = np.flatnonzero(masks[key])
            ids = (
                ids[np.linspace(0, len(ids) - 1, min(2500, len(ids)), dtype=int)]
                if len(ids)
                else ids
            )
            ax.scatter(points[ids, i], points[ids, j], s=0.4, c=color, label=key)
        ax.set(
            title=name,
            xlabel="RAS " + "XYZ"[i] + " (mm)",
            ylabel="RAS " + "XYZ"[j] + " (mm)",
            aspect="equal",
        )
    axes[0].legend(fontsize=6)
    fig.suptitle(
        "Geometry projections, not MRI slices. Unresolved sources are not declared incorrect."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
