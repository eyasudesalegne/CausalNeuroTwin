import numpy as np
import pytest

from causalneurotwin.validation.structural.anatomy import (
    cluster_overlap,
    geometry_diagnostics,
    selections,
)
from causalneurotwin.validation.structural.atlas import alternatives, mapping_evidence, support
from causalneurotwin.validation.structural.indexing import RegionIndex


def masks(d, review=()):
    return selections(
        d["original_trusted"], d["disagreement"], d["persistent"], d["template_unknown"], review
    )


@pytest.mark.parametrize(
    "ids,names",
    [
        ((1, 1), ("A_L", "A_R")),
        ((1,), ("Unknown",)),
        ((0,), ("A_L",)),
        ((1.5,), ("A_L",)),
        ((), ()),
    ],
)
def test_invalid_index(ids, names):
    with pytest.raises(ValueError):
        RegionIndex(ids, names)


def test_exact_original_order(index):
    index.check_order(index.ids)
    with pytest.raises(ValueError):
        index.check_order(index.ids[::-1])
    with pytest.raises(ValueError):
        index.check_order(index.ids[:-1])


def test_cortical_hemisphere_not_coordinate(index, data):
    trusted = index.correct_hemisphere(data["direct"], data["hemisphere"])
    assert np.array_equal(trusted, data["original_trusted"])
    assert not trusted[3] and trusted[6]


def test_no_automatic_outside_gm_or_midline_exclusion(data):
    m, status = masks(data)
    assert np.array_equal(m["original_trusted"], m["reviewed_conservative"])
    assert m["reviewed_conservative"][0] and m["reviewed_conservative"][6]
    assert not status["independent_global_acceptance"]
    assert not status["template_unknown_is_ground_truth"]


def test_uncertainty_decomposition(data):
    m, _ = masks(data)
    assert not (m["sensitivity_core"] & m["unresolved_component"]).any()
    assert np.array_equal(
        m["sensitivity_core"] | m["unresolved_component"], m["reviewed_conservative"]
    )
    assert m["unresolved_component"].sum() == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", "UNCERTAIN"),
        ("independent", False),
        ("reviewer", ""),
        ("date", ""),
        ("evidence_sha256", "bad"),
    ],
)
def test_incomplete_review_cannot_exclude(data, disposition, field, value):
    row = dict(disposition, vertex=0, classification="noncortical_closure")
    row[field] = value
    with pytest.raises(ValueError):
        masks(data, [row])


@pytest.mark.parametrize("kind", ["noncortical_closure", "invalid_source"])
def test_explicit_review_excludes_only_named_vertex(data, disposition, kind):
    m, s = masks(data, [dict(disposition, vertex=0, classification=kind)])
    assert not m["reviewed_conservative"][0] and m["reviewed_conservative"][6]
    assert s["reviewed_excluded_vertices"] == 1


def test_reviewed_cortical_not_uncertain(data, disposition):
    m, _ = masks(data, [dict(disposition, vertex=0, classification="cortical")])
    assert m["reviewed_conservative"][0] and not m["unresolved_component"][0]


def test_conflicting_review_rejected(data, disposition):
    rows = [dict(disposition, vertex=0, classification=k) for k in ["cortical", "invalid_source"]]
    with pytest.raises(ValueError):
        masks(data, rows)


def test_invalid_persistence_rejected(data):
    data["persistent"][1] = True
    with pytest.raises(ValueError):
        masks(data)


def test_geometry_reuses_areas_not_new_tissue_sampling(data):
    m, _ = masks(data)
    result = geometry_diagnostics(data["points"], data["area"], m)
    assert result["original_trusted"]["area_mm2"] == 24
    assert result["original_trusted"]["vertices"] == 6
    assert result["original_trusted"]["retention_fraction"] == 1


def test_saved_clusters_keep_hemispheres_separate(data):
    rows = cluster_overlap(
        data["cluster_id"],
        data["hemisphere"],
        data["template_unknown"],
        data["disagreement"],
        data["persistent"],
    )
    assert {(r["hemisphere"], r["saved_cluster_id"]) for r in rows} == {
        ("L", 0),
        ("L", 1),
        ("R", 0),
    }
    assert sum(r["template_unknown_overlap"] for r in rows) == 1


def test_original_direct_vs_completion(index, data, flagged):
    trusted, rows = mapping_evidence(
        index, data["direct"], data["completed"], data["distance"], data["hemisphere"], flagged
    )
    assert trusted.sum() == 6
    assert rows[1]["unaccepted_completion_vertices"] == 1
    assert not trusted[-1]


@pytest.mark.parametrize("change", ["seed", "distance", "hemisphere", "count"])
def test_invalid_saved_mapping_evidence(index, data, flagged, change):
    if change == "seed":
        data["completed"][0] = 3
    elif change == "distance":
        data["distance"][0] = 1
    elif change == "hemisphere":
        data["completed"][-1] = 1
    else:
        flagged[0]["seed_vertices"] = "10"
    with pytest.raises(ValueError):
        mapping_evidence(
            index, data["direct"], data["completed"], data["distance"], data["hemisphere"], flagged
        )


def test_atlas_ablation_keeps_full_model(index, data):
    m, _ = masks(data)
    v = alternatives(index, data["direct"], data["hemisphere"], m["reviewed_conservative"], [1, 4])
    rows, denom = support(index, *v["flagged_parcel_ablation"], data["area"])
    assert len(rows) == 5 and len(denom) == 5 and denom[-1] == 0
    assert v["reviewed_alternative"] is None
    assert not v["flagged_parcel_ablation"][1][0]


def test_alternative_requires_evidence_and_correct_hemisphere(index, data, disposition):
    m, _ = masks(data)
    args = (index, data["direct"], data["hemisphere"], m["reviewed_conservative"], [1])
    with pytest.raises(ValueError):
        alternatives(*args, [dict(vertex=0, from_id=1, to_id=3)])
    with pytest.raises(ValueError):
        alternatives(*args, [dict(disposition, vertex=0, from_id=1, to_id=2)])
    v = alternatives(*args, [dict(disposition, vertex=0, from_id=1, to_id=3)])
    assert v["reviewed_alternative"][0][0] == 3
    assert data["direct"][0] == 1


@pytest.mark.parametrize("bad", ["unknown", "deep", "negative_area", "nan_area"])
def test_support_refuses_silent_source_loss(index, data, bad):
    labels = data["direct"].copy()
    areas = data["area"].copy()
    mask = data["original_trusted"]
    if bad == "unknown":
        labels[0] = 999
    elif bad == "deep":
        labels[0] = 211
    elif bad == "negative_area":
        areas[0] = -1
    else:
        areas[0] = np.nan
    with pytest.raises(ValueError):
        support(index, labels, mask, areas)


def test_bilateral_retention_and_centroid_change(data, disposition):
    m, _ = masks(data, [dict(disposition, vertex=0, classification="invalid_source")])
    r = geometry_diagnostics(data["points"], data["area"], m, hemisphere=data["hemisphere"])
    assert r["original_trusted"]["by_hemisphere"]["L"]["vertices"] == 3
    assert r["reviewed_conservative"]["by_hemisphere"]["L"]["vertices"] == 2
    assert r["reviewed_conservative"]["by_hemisphere"]["R"]["vertices"] == 3
    assert r["reviewed_conservative"]["area_retention_fraction"] == 23 / 24
    assert r["reviewed_conservative"]["centroid_shift_from_baseline_mm"] > 0
