import sys

import numpy as np
import pytest


def test_geometry_plot_is_nonparticipant_and_exclusive(tmp_path, data):
    from causalneurotwin.validation.structural.anatomy import selections
    from causalneurotwin.validation.structural.plots import mask_figure

    m, _ = selections(
        data["original_trusted"], data["disagreement"], data["persistent"], data["template_unknown"]
    )
    path = tmp_path / "synthetic_geometry.png"
    mask_figure(path, data["points"], m)
    assert path.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        mask_figure(path, data["points"], m)


def test_template_transfer_uses_documented_direction_and_nearest(monkeypatch):
    import types

    import nibabel

    from causalneurotwin.validation.structural.anatomy import template_candidates

    calls = []

    class SyntheticSphere:
        def __init__(self, name):
            self.name = name

        @classmethod
        def from_file(cls, path):
            return cls(path)

        def project_and_resample(self, target, mask, method):
            calls.append((self.name, target.name, method))
            return mask[[2, 0, 1]]

    monkeypatch.setitem(sys.modules, "cortech", types.SimpleNamespace(Sphere=SyntheticSphere))
    monkeypatch.setattr(
        nibabel.freesurfer.io, "read_morph_data", lambda path: np.array([1, 0, 1.0])
    )
    out = template_candidates("subject_sphere", "template_sphere", "template_mask")
    assert calls == [("template_sphere", "subject_sphere", "nearest")]
    assert np.array_equal(out, [False, False, True])


def test_invalid_template_values_fail(monkeypatch):
    import types

    import nibabel

    from causalneurotwin.validation.structural.anatomy import template_candidates

    monkeypatch.setitem(sys.modules, "cortech", types.SimpleNamespace(Sphere=object))
    monkeypatch.setattr(nibabel.freesurfer.io, "read_morph_data", lambda path: np.array([0, 0.5]))
    with pytest.raises(ValueError):
        template_candidates("subject", "template", "mask")
