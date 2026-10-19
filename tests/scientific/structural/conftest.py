import numpy as np
import pytest

from causalneurotwin.validation.structural.indexing import RegionIndex


@pytest.fixture
def index():
    return RegionIndex((1, 2, 3, 4, 211), ("A_L", "A_R", "B_L", "B_R", "Deep_L"))


@pytest.fixture
def data():
    # Explicit hemispheres, including one historical wrong-side assignment.
    direct = np.array([1, 1, 3, 2, 2, 2, 4, 0], dtype=int)
    completed = np.array([1, 1, 3, 3, 2, 2, 4, 4], dtype=int)
    trusted = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    passed = np.zeros((5, 5))
    lengths = np.zeros((5, 5))
    for i, j, w, l in [(0, 1, 4, 10), (1, 2, 2, 20), (2, 3, 3, 30), (3, 4, 1, 40)]:
        passed[i, j] = passed[j, i] = w
        lengths[i, j] = lengths[j, i] = l
    endpoint = passed.copy()
    endpoint[1, 2] = endpoint[2, 1] = 0
    el = lengths.copy()
    el[1, 2] = el[2, 1] = 0
    rows = []
    ids = [1, 2, 3, 4, 211]
    for i in range(5):
        for j in range(i + 1, 5):
            rows.append(
                dict(
                    atlas_i=str(ids[i]),
                    atlas_j=str(ids[j]),
                    end_count=str(endpoint[i, j]),
                    pass_count=str(passed[i, j]),
                    end_length_mm=str(el[i, j]) if endpoint[i, j] else "",
                    pass_length_mm=str(lengths[i, j]) if passed[i, j] else "",
                )
            )
    return dict(
        points=np.arange(24, dtype=float).reshape(8, 3),
        area=np.arange(1, 9, dtype=float),
        direct=direct,
        completed=completed,
        distance=np.array([0, 0, 0, 2, 0, 0, 0, 3.0]),
        hemisphere=np.array(["L"] * 4 + ["R"] * 4),
        original_trusted=trusted,
        disagreement=np.array([1, 0, 0, 1, 0, 0, 1, 0], bool),
        persistent=np.array([1, 0, 0, 0, 0, 0, 1, 0], bool),
        template_unknown=np.array([1, 0, 0, 0, 0, 0, 0, 0], bool),
        cluster_id=np.array([0, -1, -1, 1, -1, -1, 0, -1]),
        edge_rows=rows,
        pass_reference=passed / 6.0,
        length_reference=lengths,
        nulls=[],
    )


@pytest.fixture
def flagged():
    return [
        dict(atlas_id="1", label="A_L", seed_vertices="2"),
        dict(atlas_id="4", label="B_R", seed_vertices="1"),
    ]


@pytest.fixture
def disposition():
    return dict(
        decision="ACCEPT",
        independent=True,
        reviewer="Synthetic reviewer",
        date="2026-10-18",
        evidence="synthetic-only.txt",
        evidence_sha256="a" * 64,
    )
