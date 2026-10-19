import numpy as np
import pytest

from causalneurotwin.validation.structural.connectome import (
    compare,
    from_saved_edges,
    network_metrics,
    validate,
    variants,
)
from causalneurotwin.validation.structural.pipeline import evaluate


def nets(index, data):
    e, p, el, pl = from_saved_edges(index, data["edge_rows"])
    return e, p, el, pl


def test_fixed_reference_normalization_and_own_lengths(index, data):
    e, p, el, pl = nets(index, data)
    v = variants(index, e, p, el, pl, 6.0)
    assert np.array_equal(v["pass"]["weights"], p / 6.0)
    assert np.array_equal(v["endpoint"]["weights"], e / 6.0)
    assert v["endpoint"]["weights"].sum() < v["pass"]["weights"].sum()
    assert np.array_equal(v["endpoint"]["lengths"], el)


def test_common_support_uses_pass_weights_and_lengths(index, data):
    e, p, el, pl = nets(index, data)
    e[0, 1] = e[1, 0] = 2
    el[0, 1] = el[1, 0] = 12
    v = variants(index, e, p, el, pl, 6.0)
    common = v["common_pass_weighted"]
    assert common["weights"][0, 1] == 4 / 6.0
    assert common["lengths"][0, 1] == 10
    assert common["weights"][1, 2] == 0 and common["lengths"][1, 2] == 0


@pytest.mark.parametrize(
    "kind", ["shape", "nan", "asym", "diag", "missinglength", "absentlength", "negative", "order"]
)
def test_network_guards(index, data, kind):
    _e, p, _el, pl = nets(index, data)
    ids = index.ids
    if kind == "shape":
        p = p[:-1, :-1]
    elif kind == "nan":
        p[0, 1] = np.nan
    elif kind == "asym":
        p[0, 1] += 1
    elif kind == "diag":
        p[0, 0] = 1
        pl[0, 0] = 1
    elif kind == "missinglength":
        pl[0, 1] = pl[1, 0] = 0
    elif kind == "absentlength":
        pl[0, 4] = pl[4, 0] = 1
    elif kind == "negative":
        p[0, 1] = p[1, 0] = -1
    else:
        ids = ids[::-1]
    with pytest.raises(ValueError):
        validate(index, p, pl, ids)


@pytest.mark.parametrize("scale", [0, -1, 5, np.nan])
def test_wrong_normalization_fails(index, data, scale):
    with pytest.raises(ValueError):
        variants(index, *nets(index, data), scale)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown"])
def test_saved_edge_table_complete(index, data, kind):
    rows = data["edge_rows"]
    if kind == "missing":
        rows = rows[:-1]
    elif kind == "duplicate":
        rows = rows + [rows[0]]
    else:
        rows[0]["atlas_i"] = "999"
    with pytest.raises(ValueError):
        from_saved_edges(index, rows)


def null_fixture(index, data):
    e, p, el, pl = nets(index, data)
    # Synthetic preservation fixture only; not a newly generated scientific null.
    L = pl.copy()
    L[0, 1] = L[1, 0] = 20
    L[1, 2] = L[2, 1] = 10
    return (
        e,
        p,
        el,
        pl,
        dict(
            id="null_01",
            weights=p / 6.0,
            lengths=L,
            ids=index.ids,
            units="original_pass_region_scaled",
        ),
    )


def test_existing_null_never_double_normalized(index, data):
    e, p, el, pl, n = null_fixture(index, data)
    v = variants(index, e, p, el, pl, 6.0, [n])
    assert np.array_equal(v["null_01"]["weights"], p / 6.0)
    assert v["null_01"]["lengths"][0, 1] != pl[0, 1]
    assert "pairings changed" in v["null_01"]["role"]


@pytest.mark.parametrize("kind", ["units", "degree", "strength", "length", "count"])
def test_null_provenance_claim_guards(index, data, kind):
    e, p, el, pl, n = null_fixture(index, data)
    if kind == "units":
        n["units"] = "raw_count"
    elif kind == "degree":
        n["weights"][0, 1] = n["weights"][1, 0] = 0
        n["lengths"][0, 1] = n["lengths"][1, 0] = 0
    elif kind == "strength":
        n["weights"] *= 2
    elif kind == "length":
        n["lengths"][0, 1] = n["lengths"][1, 0] = 99
    with pytest.raises(ValueError):
        variants(index, e, p, el, pl, 6.0, [n] * 21 if kind == "count" else [n])


def test_metrics_delays_and_topology(index, data):
    v = variants(index, *nets(index, data), 6.0)
    m = network_metrics(index, v["pass"], speed=4.0)
    assert m["edges"] == 4 and m["components"] == 1
    assert m["degree"] == [1, 2, 2, 2, 1]
    assert m["delay_ms_distribution"]["max"] == 10
    c = compare(index, v["pass"], v["endpoint"])
    assert c["jaccard"] == 0.75 and not c["secondary_factor_applied"]


def test_joint_pipeline_pure_synthetic(index, data, flagged):
    before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    out = evaluate(index, data, flagged, {}, 6.0)
    assert out["report"]["EEG_prediction_performed"] is False
    assert out["report"]["independent_anatomical_acceptance"] is False
    assert len(out["networks"]) == 3
    assert len(out["report"]["atlas_variants"]["original_direct"]) == 5
    for k, v in before.items():
        assert np.array_equal(v, data[k])


@pytest.mark.parametrize("kind", ["trusted", "weights", "lengths"])
def test_pipeline_frozen_baseline_guard(index, data, flagged, kind):
    if kind == "trusted":
        data["original_trusted"][0] = False
    elif kind == "weights":
        data["pass_reference"] *= 2
    else:
        data["length_reference"][0, 1] += 1
    with pytest.raises(ValueError):
        evaluate(index, data, flagged, {}, 6.0)
