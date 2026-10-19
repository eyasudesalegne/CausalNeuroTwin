import json
from pathlib import Path

import pytest

from causalneurotwin.experiment_records import validate_experiment_record


def draft():
    p = Path(__file__).resolve().parents[1] / "configs/experiment_record.example.json"
    return json.loads(p.read_text())


def complete():
    r = draft()
    r.update(
        status="completed",
        git_commit="a" * 40,
        git_dirty=False,
        config_version="synthetic-v1",
        config_sha256="b" * 64,
        code_manifest_sha256="c" * 64,
        input_integrity_manifest_sha256="d" * 64,
        result_manifest_sha256="e" * 64,
        slurm_job_id="123",
        software_versions={"python": "synthetic"},
        linkage="verified_at_execution",
        disclosure_review={
            "approved": True,
            "reviewer": "SYNTHETIC",
            "date": "2026-01-01",
            "scope": "nonparticipant fixture",
        },
    )
    return r


def test_draft_does_not_invent_links_or_approval():
    r = draft()
    assert validate_experiment_record(r)["git_commit"] is None
    assert not r["disclosure_review"]["approved"]


def test_complete_synthetic_record_is_fully_linked():
    assert validate_experiment_record(complete())["status"] == "completed"


@pytest.mark.parametrize(
    "key",
    [
        "git_commit",
        "config_sha256",
        "code_manifest_sha256",
        "input_integrity_manifest_sha256",
        "slurm_job_id",
        "result_manifest_sha256",
    ],
)
def test_missing_link_blocks_completion(key):
    r = complete()
    r[key] = None
    with pytest.raises(ValueError):
        validate_experiment_record(r)


@pytest.mark.parametrize("change", ["dirty", "retrospective", "disclosure"])
def test_history_and_disclosure_not_fabricated(change):
    r = complete()
    if change == "dirty":
        r["git_dirty"] = True
    elif change == "retrospective":
        r["linkage"] = "retrospective_source_hash_only"
    else:
        r["disclosure_review"]["approved"] = False
    with pytest.raises(ValueError):
        validate_experiment_record(r)


def test_unreviewed_internal_scheduler_metadata_rejected():
    r = draft()
    r["slurm_job_id"] = "123"
    with pytest.raises(ValueError):
        validate_experiment_record(r)
