"""Joint new uncertainty diagnostics from saved arrays; no reconstructions or simulations."""

import numpy as np

from .anatomy import cluster_overlap, geometry_diagnostics, selections
from .atlas import alternatives, mapping_evidence, support
from .connectome import compare, from_saved_edges, network_metrics, variants


def evaluate(index, data, flagged, review, reference_scale):
    trusted, parcels = mapping_evidence(
        index, data["direct"], data["completed"], data["distance"], data["hemisphere"], flagged
    )
    if not np.array_equal(trusted, data["original_trusted"]):
        raise ValueError("Original trusted mask not reproduced")
    masks, mask_status = selections(
        trusted,
        data["disagreement"],
        data["persistent"],
        data["template_unknown"],
        review.get("exclusions", []),
    )
    geometry = geometry_diagnostics(
        data["points"], data["area"], masks, hemisphere=data["hemisphere"]
    )
    mapping = alternatives(
        index,
        data["direct"],
        data["hemisphere"],
        masks["reviewed_conservative"],
        [int(x["atlas_id"]) for x in flagged],
        review.get("reassignments", []),
    )
    _baseline_rows, baseline_area = support(index, data["direct"], trusted, data["area"])
    atlas_summary = {}
    for key, value in mapping.items():
        atlas_summary[key] = (
            None
            if value is None
            else support(index, value[0], value[1], data["area"], baseline_area)[0]
        )
    end, passed, el, pl = from_saved_edges(index, data["edge_rows"])
    if not np.allclose(passed / reference_scale, data["pass_reference"], atol=1e-15, rtol=1e-12):
        raise ValueError("Saved pass weights no longer match frozen model")
    if not np.array_equal(pl, data["length_reference"]):
        raise ValueError("Pass length definition differs")
    networks = variants(index, end, passed, el, pl, reference_scale, data.get("nulls", []))
    stats = {name: network_metrics(index, v) for name, v in networks.items()}
    comparisons = {
        name: compare(index, networks["pass"], v) for name, v in networks.items() if name != "pass"
    }
    report = {
        "anatomy": mask_status,
        "geometry": geometry,
        "cluster_overlap": cluster_overlap(
            data["cluster_id"],
            data["hemisphere"],
            data["template_unknown"],
            data["disagreement"],
            data["persistent"],
        ),
        "flagged_parcels": parcels,
        "atlas_variants": atlas_summary,
        "network_metrics": stats,
        "network_comparisons": comparisons,
        "independent_anatomical_acceptance": False,
        "EEG_prediction_performed": False,
        "source_mapping_does_not_change_neural_connectivity": True,
        "DWI_acquisition_uncertainty_recovered": False,
    }
    return {
        "masks": masks,
        "mapping": mapping,
        "networks": networks,
        "report": report,
        "baseline_area": baseline_area,
    }
