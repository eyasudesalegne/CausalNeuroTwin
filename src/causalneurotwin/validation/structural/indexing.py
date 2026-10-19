"""Explicit immutable model order; noncortical nodes never silently disappear."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionIndex:
    ids: tuple
    names: tuple

    def __post_init__(self):
        if not self.ids or len(self.ids) != len(self.names) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Unique atlas IDs and one name per model node required")
        if any(not isinstance(x, (int, np.integer)) or x <= 0 for x in self.ids):
            raise ValueError("Positive integer atlas IDs")
        for aid, name in zip(self.ids, self.names):
            if aid <= 210 and not name.endswith(("_L", "_R")):
                raise ValueError("Cortical hemisphere label required")

    @property
    def n(self):
        return len(self.ids)

    @property
    def cortical(self):
        return np.array([x <= 210 for x in self.ids])

    def check_order(self, ids):
        if not np.array_equal(np.asarray(ids), self.ids):
            raise ValueError("Atlas order differs; no automatic reordering")

    def check_square(self, array):
        if np.shape(array) != (self.n, self.n):
            raise ValueError("Model node count/layout mismatch")

    def correct_hemisphere(self, labels, hemisphere):
        labels = np.asarray(labels)
        hemisphere = np.asarray(hemisphere)
        if (
            labels.ndim != 1
            or labels.shape != hemisphere.shape
            or not np.isin(hemisphere, ["L", "R"]).all()
        ):
            raise ValueError("Explicit hemisphere identity per source required")
        result = np.zeros(len(labels), bool)
        for aid, name in zip(self.ids, self.names):
            if aid <= 210:
                result |= (labels == aid) & (hemisphere == name[-1])
        return result
