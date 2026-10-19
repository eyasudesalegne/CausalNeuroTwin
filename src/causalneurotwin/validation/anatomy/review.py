"""Reviewer-attested assessments, independent landmark inputs and reused parcel triage."""

from datetime import datetime
from pathlib import Path

import numpy as np

from .geometry import apply, point_map
from .io import CONSISTENCY, INDEPENDENT, PENDING, digest, inside, read_csv, write_csv

REVIEW_FIELDS = [
    "item_id",
    "domain",
    "decision",
    "reason",
    "evidence_image",
    "reviewer_id",
    "reviewed_at_utc",
    "independence_basis",
    "independent_reference",
]
LANDMARK_FIELDS = [
    "landmark_id",
    "source_image",
    "target_image",
    "source_x_mm",
    "source_y_mm",
    "source_z_mm",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "coordinate_system",
    "reviewer_id",
    "independence_basis",
    "source_evidence",
    "target_evidence",
]
CHECKLIST = [
    ("orientation_lr", "orientation"),
    ("landmark_ac_pc", "landmarks"),
    ("midline_falx", "landmarks"),
    ("wm_gm_left", "tissues"),
    ("wm_gm_right", "tissues"),
    ("gm_csf_sulci", "tissues"),
    ("ventricles_csf", "tissues"),
    ("inner_skull", "skull"),
    ("outer_skull", "skull"),
    ("medial_wall_left", "surfaces"),
    ("medial_wall_right", "surfaces"),
    ("insula_left", "surfaces"),
    ("insula_right", "surfaces"),
    ("temporal_left", "surfaces"),
    ("temporal_right", "surfaces"),
    ("amygdala_hippocampus_left", "deep_structures"),
    ("amygdala_hippocampus_right", "deep_structures"),
    ("thalamus_basal_ganglia", "deep_structures"),
    ("cerebellum_brainstem", "deep_structures"),
    ("registration_local_alignment", "registration"),
    ("surface_tissue_candidates", "boundary_candidates"),
    ("mesh_shape_candidates", "mesh"),
]


def parcel_queue(artifacts, thresholds):
    """Reuses stored labels/counts; NEVER resamples or repairs atlas labels."""
    counts = read_csv(artifacts["surface_counts"]["path"], delimiter="\t")
    distort = {
        int(r["atlas_id"]): r
        for r in read_csv(artifacts["parcel_distortion"]["path"], delimiter="\t")
    }
    rows = []
    for row in counts:
        atlas_id = int(row["atlas_id"])
        if not 1 <= atlas_id <= 210:
            continue
        if atlas_id not in distort:
            raise ValueError(f"Missing pre-existing cortical distortion row: {atlas_id}")
        d = distort[atlas_id]
        seed, total = int(d["seed_vertices"]), int(d["final_vertices"])
        fraction = seed / total if total else 0
        wrong = (
            int(row["surface_vertices_rh"])
            if row["label"].endswith("_L")
            else int(row["surface_vertices_lh"])
            if row["label"].endswith("_R")
            else 0
        )
        reasons = []
        if fraction < thresholds["trusted_fraction_below"]:
            reasons.append("low_trusted_fraction")
        if wrong:
            reasons.append("recorded_wrong_hemisphere_samples")
        if float(d["expansion_ratio"]) > thresholds["expansion_ratio_above"]:
            reasons.append("extensive_geodesic_completion")
        if int(d["added_gt10mm"]) > thresholds["added_gt10mm_above"]:
            reasons.append("long_distance_completion")
        rows.append(
            {
                "atlas_id": atlas_id,
                "label": row["label"],
                "seed_vertices": seed,
                "final_vertices": total,
                "trusted_fraction": fraction,
                "wrong_hemisphere_samples": wrong,
                "expansion_ratio": float(d["expansion_ratio"]),
                "added_gt10mm": int(d["added_gt10mm"]),
                "triage_reasons": ";".join(reasons),
                "evidence_image": "",
                "evidence_category": PENDING,
                "assessment_status": "pending",
            }
        )
    if len(rows) != 210 or len({r["atlas_id"] for r in rows}) != 210:
        raise ValueError("Pre-existing parcel-table coverage is not 210 unique cortical regions")
    return sorted(
        rows,
        key=lambda r: (
            not bool(r["triage_reasons"]),
            -r["expansion_ratio"],
            -r["wrong_hemisphere_samples"],
        ),
    )


def landmark_errors(path, volumes, pull, triage_mm):
    fields = [
        "landmark_id",
        "source_image",
        "target_image",
        "error_mm",
        "triage_above_mm",
        "reviewer_id",
        "evidence_category",
        "assessment_status",
    ]
    rows, points, incomplete = [], [], []
    if not path:
        return (
            {
                "evidence_category": PENDING,
                "assessment_status": "pending",
                "reason": "Independent landmark annotations have not been supplied",
                "n_complete": 0,
                "triage_threshold_mm_provisional": triage_mm,
            },
            rows,
            fields,
            points,
        )
    source_rows = read_csv(path)
    seen = set()
    for row in source_rows:
        lid = row.get("landmark_id", "").strip()
        if not lid or lid in seen:
            raise ValueError("Landmark IDs must be nonempty and unique")
        seen.add(lid)
        if any(not row.get(k, "").strip() for k in LANDMARK_FIELDS):
            incomplete.append(lid)
            continue
        if row["coordinate_system"] != "RAS_mm":
            raise ValueError("Landmark coordinates must explicitly be RAS_mm")
        if row["independence_basis"] != "separately_identified_not_transformed":
            raise ValueError("Reject landmarks copied/transformed from the other image")
        sk, tk = row["source_image"], row["target_image"]
        if sk not in volumes or tk not in volumes or sk == tk:
            raise ValueError("Landmark source/target must be distinct configured images")
        s = np.array([float(row[f"source_{a}_mm"]) for a in "xyz"])
        t = np.array([float(row[f"target_{a}_mm"]) for a in "xyz"])
        if not np.isfinite(s).all() or not np.isfinite(t).all():
            raise ValueError("Nonfinite landmark")
        if not volumes[sk].contains(s) or not volumes[tk].contains(t):
            raise ValueError(f"{lid}: annotation outside its declared image FOV")
        m = point_map(volumes[sk].space, volumes[tk].space, pull)
        error = float(np.linalg.norm(apply(m, s) - t))
        rows.append(
            {
                "landmark_id": lid,
                "source_image": sk,
                "target_image": tk,
                "error_mm": error,
                "triage_above_mm": error > triage_mm,
                "reviewer_id": row["reviewer_id"],
                "evidence_category": CONSISTENCY,
                "assessment_status": "measured",
            }
        )
        points.append(
            (
                lid,
                apply(point_map(volumes[sk].space, "native", pull), s),
                apply(point_map(volumes[tk].space, "native", pull), t),
            )
        )
    report = {
        "evidence_category": CONSISTENCY if rows else PENDING,
        "assessment_status": "measured" if rows else "pending",
        "independent_anatomical_assessment": "pending",
        "n_complete": len(rows),
        "incomplete_landmark_ids": incomplete,
        "triage_threshold_mm_provisional": triage_mm,
        "source_annotations_sha256": digest(path),
        "limitation": "Independent manual identification is an assessment of alignment; "
        "same-source derivatives remain nonindependent imaging ground truth.",
    }
    if rows:
        a = np.array([r["error_mm"] for r in rows])
        report.update(
            median_error_mm=float(np.median(a)),
            p95_error_mm=float(np.quantile(a, 0.95)),
            max_error_mm=float(a.max()),
        )
    return report, rows, fields, points


def review_items(parcels):
    return [{"item_id": k, "domain": d} for k, d in CHECKLIST] + [
        {"item_id": f"parcel_{r['atlas_id']}", "domain": "parcellation"}
        for r in parcels
        if r["triage_reasons"]
    ]


def write_review_template(output, items, suggested):
    rows = []
    for item in items:
        rows.append(
            {
                **{k: "" for k in REVIEW_FIELDS},
                **item,
                "decision": "pending",
                "evidence_image": suggested.get(item["item_id"], ""),
                "evidence_category": PENDING,
                "assessment_status": "pending",
            }
        )
    write_csv(
        output / "review_annotations.csv",
        rows,
        REVIEW_FIELDS + ["evidence_category", "assessment_status"],
    )


def assess_review(path, items, bundle, evidence_index):
    """Attest scoped reviewer decisions; never turn numerical thresholds into anatomy passes."""
    supplied = read_csv(path) if path else []
    indexed = {}
    for r in supplied:
        key = r.get("item_id")
        if key in indexed or key not in {i["item_id"] for i in items}:
            raise ValueError("Duplicate or unknown review item")
        indexed[key] = r
    results = []
    for item in items:
        row = indexed.get(item["item_id"], {})
        decision = row.get("decision", "pending").strip()
        if decision not in {"accept", "reject", "uncertain", "pending"}:
            raise ValueError("Review decision must be accept/reject/uncertain/pending")
        problems = []
        if row and row.get("domain") != item["domain"]:
            problems.append("domain_mismatch")
        image = row.get("evidence_image", "")
        if decision in {"accept", "reject", "uncertain"}:
            for key in ["reason", "reviewer_id", "reviewed_at_utc", "independence_basis"]:
                if not row.get(key, "").strip():
                    problems.append("missing_" + key)
            try:
                stamp = datetime.fromisoformat(
                    row.get("reviewed_at_utc", "").replace("Z", "+00:00")
                )
                if stamp.utcoffset() is None:
                    problems.append("timestamp_requires_timezone")
            except ValueError:
                problems.append("invalid_timestamp")
            if image not in evidence_index:
                problems.append("unknown_evidence_image")
            else:
                target = Path(bundle) / image
                if (
                    not inside(target, bundle)
                    or not target.is_file()
                    or digest(target) != evidence_index[image]
                ):
                    problems.append("evidence_missing_or_changed")
            if item["item_id"] == "orientation_lr" and (
                row.get("independence_basis") != "acquisition_side_reference"
                or not row.get("independent_reference", "").strip()
            ):
                problems.append("left_right_requires_independent_acquisition_side_reference")
            if item["domain"] == "mesh":
                if row.get("independence_basis") != "technical_mesh_review":
                    problems.append("mesh_item_requires_technical_review_scope")
            elif item["item_id"] != "orientation_lr" and (
                row.get("independence_basis") != "independently_inspected_original_T1"
            ):
                problems.append("requires_independent_original_T1_assessment")
        accepted = decision == "accept" and not problems
        technical = item["domain"] == "mesh"
        results.append(
            {
                **item,
                "decision": decision,
                "assessment_status": ("technical_reviewed" if technical else "passed")
                if accepted
                else "failed"
                if decision == "reject" and not problems
                else "indeterminate"
                if problems or decision == "uncertain"
                else "pending",
                "evidence_category": (CONSISTENCY if technical else INDEPENDENT)
                if accepted
                else PENDING,
                "independent_anatomical_assessment": "pending"
                if technical
                else "passed_for_this_item_only"
                if accepted
                else "pending",
                "reason": row.get("reason", ""),
                "reviewer_id": row.get("reviewer_id", ""),
                "reviewed_at_utc": row.get("reviewed_at_utc", ""),
                "independence_basis": row.get("independence_basis", ""),
                "independent_reference": row.get("independent_reference", ""),
                "evidence_image": image,
                "record_problems": problems,
            }
        )
    return {
        "evidence_category": PENDING,
        "assessment_status": "review_recorded" if supplied else "pending",
        "scope": "Item-level reviewer-attested macroscopic anatomical assessment only",
        "whole_subject_biological_validation": "not_established",
        "passed_items": sum(r["assessment_status"] == "passed" for r in results),
        "technical_reviewed_items": sum(
            r["assessment_status"] == "technical_reviewed" for r in results
        ),
        "required_items": len(results),
        "items": results,
    }
