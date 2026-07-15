from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from causalneurotwin.datasets import (
    DatasetRegistryEntry,
    DatasetRegistryError,
    load_dataset_registry,
    resolve_dataset_root,
    validate_local_dataset,
    write_dataset_validation_report,
)

PARTICIPANTS = (
    "sub-CON001",
    "sub-CON006",
    "sub-CON008",
    "sub-CON009",
    "sub-CON015",
    "sub-CON019",
    "sub-CON020",
    "sub-CON021",
    "sub-CON022",
    "sub-CON025",
    "sub-CON026",
    "sub-CON029",
    "sub-CON031",
)


def create_dataset(root: Path) -> None:
    root.mkdir()
    description = {
        "Name": (
            "TMS-EEG-MRI-fMRI-DWI data on paired associative stimulation and connectivity "
            "(Shirley Ryan AbilityLab, Chicago, IL)"
        ),
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "License": "CC0",
        "DatasetDOI": "doi:10.18112/openneuro.ds004024.v1.0.1",
    }
    (root / "dataset_description.json").write_text(json.dumps(description), encoding="utf-8")
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\thand\n"
        + "".join(f"{participant}\tn/a\tn/a\tn/a\n" for participant in PARTICIPANTS),
        encoding="utf-8",
    )
    (root / "participants.json").write_text("{}\n", encoding="utf-8")
    (root / "README").write_text("fixture\n", encoding="utf-8")
    (root / "CHANGES").write_text("fixture\n", encoding="utf-8")
    for index, participant in enumerate(PARTICIPANTS):
        subject = root / participant
        (subject / "anat").mkdir(parents=True)
        (subject / "eeg").mkdir()
        (subject / "anat" / f"{participant}_T1w.nii.gz").touch()
        if index % 2 == 0:
            (subject / "func").mkdir()
        if index % 3 == 0:
            (subject / "dwi").mkdir()


@pytest.fixture
def registry_path() -> Path:
    return Path("configs/data/openneuro_ds004024.yaml")


@pytest.fixture
def registry(registry_path: Path) -> DatasetRegistryEntry:
    return load_dataset_registry(registry_path)


def test_registry_loads_pinned_identity(registry: DatasetRegistryEntry) -> None:
    assert registry.dataset_id == "ds004024"
    assert registry.dataset_version == "1.0.1"
    assert registry.source.doi == "10.18112/openneuro.ds004024.v1.0.1"
    assert registry.expected.participant_count == 13
    assert registry.license.participant_level_data_in_git is False


def test_valid_fixture_passes(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root = tmp_path / "private-dataset-root"
    create_dataset(dataset_root)
    result = validate_local_dataset(registry, dataset_root, root_source="explicit")
    assert result.status == "pass"
    assert result.participant_count == 13
    assert result.scientific_data_ready is False
    assert sum(item.func_present for item in result.inventory) == 7
    assert str(dataset_root) not in json.dumps(result.to_dict())


def test_report_contains_no_absolute_dataset_path(
    registry: DatasetRegistryEntry, registry_path: Path, tmp_path: Path
) -> None:
    dataset_root = tmp_path / "private" / "ds004024"
    dataset_root.parent.mkdir()
    create_dataset(dataset_root)
    result = validate_local_dataset(registry, dataset_root, root_source="explicit")
    output = tmp_path / "reports"
    write_dataset_validation_report(
        registry=registry,
        validation=result,
        dataset_root=dataset_root,
        registry_path=registry_path,
        output_dir=output,
    )
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.is_file()
    )
    assert str(dataset_root) not in combined
    assert "n/a" not in combined
    assert (output / "checksums.sha256").is_file()
    assert (output / "source_metadata_checksums.sha256").is_file()


def test_mismatched_doi_fails(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    create_dataset(dataset_root)
    description_path = dataset_root / "dataset_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["DatasetDOI"] = "doi:10.18112/openneuro.ds004024.v9.9.9"
    description_path.write_text(json.dumps(description), encoding="utf-8")
    result = validate_local_dataset(registry, dataset_root, root_source="explicit")
    assert result.status == "fail"
    assert any(
        item["name"] == "dataset_description:DatasetDOI" and item["status"] == "fail"
        for item in result.checks
    )


def test_missing_subject_directory_fails(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    create_dataset(dataset_root)
    subject = dataset_root / PARTICIPANTS[-1]
    for child in sorted(subject.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
        else:
            child.unlink()
    subject.rmdir()
    result = validate_local_dataset(registry, dataset_root, root_source="explicit")
    assert result.status == "fail"


def test_missing_t1w_layout_fails(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    create_dataset(dataset_root)
    next((dataset_root / PARTICIPANTS[0] / "anat").glob("*_T1w.nii.gz")).unlink()
    result = validate_local_dataset(registry, dataset_root, root_source="explicit")
    assert result.status == "fail"


def test_environment_root_resolution(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root, source = resolve_dataset_root(
        registry,
        explicit_root=None,
        environment={"CAUSALNEUROTWIN_DATA_ROOT": str(tmp_path)},
    )
    assert dataset_root == tmp_path / "openneuro/ds004024-v1.0.1"
    assert source == "environment"


def test_missing_environment_is_reported(registry: DatasetRegistryEntry) -> None:
    with pytest.raises(DatasetRegistryError, match="CAUSALNEUROTWIN_DATA_ROOT"):
        resolve_dataset_root(registry, explicit_root=None, environment={})


def test_registry_rejects_unknown_key(registry_path: Path) -> None:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    payload["invented"] = True
    with pytest.raises(DatasetRegistryError, match="unknown keys"):
        DatasetRegistryEntry.from_mapping(payload)


def test_registry_rejects_unsafe_local_path(registry_path: Path) -> None:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    payload["access"]["relative_path"] = "../private"
    with pytest.raises(DatasetRegistryError, match="safe relative path"):
        DatasetRegistryEntry.from_mapping(payload)


def test_invalid_participant_id_is_rejected(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    create_dataset(dataset_root)
    participants = (dataset_root / "participants.tsv").read_text(encoding="utf-8")
    (dataset_root / "participants.tsv").write_text(
        participants.replace("sub-CON001", "participant-name", 1), encoding="utf-8"
    )
    with pytest.raises(DatasetRegistryError, match="invalid participant_id"):
        validate_local_dataset(registry, dataset_root, root_source="explicit")


def test_missing_dataset_root_is_rejected(registry: DatasetRegistryEntry, tmp_path: Path) -> None:
    with pytest.raises(DatasetRegistryError, match="dataset root"):
        validate_local_dataset(registry, tmp_path / "missing", root_source="explicit")
