"""Public-dataset registration and non-sensitive local-copy validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from causalneurotwin.checksums import sha256_file, write_checksum_manifest
from causalneurotwin.serialization import write_json

_REGISTRY_SCHEMA_VERSION = "1.0"
_SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9]+$")


class DatasetRegistryError(ValueError):
    """Raised when registry metadata or a local dataset copy is invalid."""


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetRegistryError(f"{location} must be a mapping")
    return cast(dict[str, Any], value)


def _keys(
    payload: dict[str, Any], *, required: set[str], optional: set[str], location: str
) -> None:
    missing = sorted(required - payload.keys())
    unknown = sorted(payload.keys() - required - optional)
    if missing:
        raise DatasetRegistryError(f"{location} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise DatasetRegistryError(f"{location} contains unknown keys: {', '.join(unknown)}")


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetRegistryError(f"{location} must be a non-empty string")
    return value.strip()


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetRegistryError(f"{location} must be a boolean")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatasetRegistryError(f"{location} must be an integer >= {minimum}")
    return value


def _strings(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetRegistryError(f"{location} must be a list")
    result = tuple(_string(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise DatasetRegistryError(f"{location} must not contain duplicates")
    return result


def _relative_path(value: object, location: str) -> str:
    result = _string(value, location)
    candidate = Path(result)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DatasetRegistryError(f"{location} must be a safe relative path")
    return candidate.as_posix()


@dataclass(frozen=True)
class DatasetSource:
    repository: str
    landing_page: str
    version_page: str
    doi: str
    mirror_repository: str

    @classmethod
    def from_mapping(cls, value: object) -> DatasetSource:
        payload = _mapping(value, "source")
        required = {"repository", "landing_page", "version_page", "doi", "mirror_repository"}
        _keys(payload, required=required, optional=set(), location="source")
        return cls(**{key: _string(payload[key], f"source.{key}") for key in required})


@dataclass(frozen=True)
class DatasetLicense:
    identifier: str
    redistributable: bool
    participant_level_data_in_git: bool

    @classmethod
    def from_mapping(cls, value: object) -> DatasetLicense:
        payload = _mapping(value, "license")
        required = {"identifier", "redistributable", "participant_level_data_in_git"}
        _keys(payload, required=required, optional=set(), location="license")
        return cls(
            identifier=_string(payload["identifier"], "license.identifier"),
            redistributable=_boolean(payload["redistributable"], "license.redistributable"),
            participant_level_data_in_git=_boolean(
                payload["participant_level_data_in_git"],
                "license.participant_level_data_in_git",
            ),
        )


@dataclass(frozen=True)
class DatasetAccess:
    access_type: Literal["public", "registered", "controlled"]
    authentication_required: bool
    environment_variable: str
    relative_path: str

    @classmethod
    def from_mapping(cls, value: object) -> DatasetAccess:
        payload = _mapping(value, "access")
        required = {
            "access_type",
            "authentication_required",
            "environment_variable",
            "relative_path",
        }
        _keys(payload, required=required, optional=set(), location="access")
        access_type = _string(payload["access_type"], "access.access_type")
        if access_type not in {"public", "registered", "controlled"}:
            raise DatasetRegistryError(
                "access.access_type must be public, registered, or controlled"
            )
        environment_variable = _string(
            payload["environment_variable"], "access.environment_variable"
        )
        if (
            not environment_variable.replace("_", "").isalnum()
            or not environment_variable.isupper()
        ):
            raise DatasetRegistryError(
                "access.environment_variable must be an uppercase environment-variable name"
            )
        return cls(
            access_type=cast(Literal["public", "registered", "controlled"], access_type),
            authentication_required=_boolean(
                payload["authentication_required"], "access.authentication_required"
            ),
            environment_variable=environment_variable,
            relative_path=_relative_path(payload["relative_path"], "access.relative_path"),
        )


@dataclass(frozen=True)
class BidsContract:
    version: str
    dataset_type: Literal["raw", "derivative"]

    @classmethod
    def from_mapping(cls, value: object) -> BidsContract:
        payload = _mapping(value, "bids")
        _keys(payload, required={"version", "dataset_type"}, optional=set(), location="bids")
        dataset_type = _string(payload["dataset_type"], "bids.dataset_type")
        if dataset_type not in {"raw", "derivative"}:
            raise DatasetRegistryError("bids.dataset_type must be raw or derivative")
        return cls(
            version=_string(payload["version"], "bids.version"),
            dataset_type=cast(Literal["raw", "derivative"], dataset_type),
        )


@dataclass(frozen=True)
class ExpectedDataset:
    participant_count: int
    top_level_files: tuple[str, ...]
    all_participants_require_t1w: bool

    @classmethod
    def from_mapping(cls, value: object) -> ExpectedDataset:
        payload = _mapping(value, "expected")
        required = {"participant_count", "top_level_files", "all_participants_require_t1w"}
        _keys(payload, required=required, optional=set(), location="expected")
        files = _strings(payload["top_level_files"], "expected.top_level_files")
        for item in files:
            if Path(item).name != item:
                raise DatasetRegistryError("expected.top_level_files must contain file names only")
        return cls(
            participant_count=_integer(
                payload["participant_count"], "expected.participant_count", minimum=1
            ),
            top_level_files=files,
            all_participants_require_t1w=_boolean(
                payload["all_participants_require_t1w"],
                "expected.all_participants_require_t1w",
            ),
        )


@dataclass(frozen=True)
class DatasetRegistryEntry:
    schema_version: str
    dataset_id: str
    dataset_version: str
    name: str
    description: str
    source: DatasetSource
    license: DatasetLicense
    access: DatasetAccess
    bids: BidsContract
    modalities: tuple[str, ...]
    optional_per_subject_modalities: tuple[str, ...]
    expected: ExpectedDataset
    intended_use: str
    prohibited_use: str
    acknowledgement: str
    ethics_approval: str

    @classmethod
    def from_mapping(cls, value: object) -> DatasetRegistryEntry:
        payload = _mapping(value, "dataset registry")
        required = {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "name",
            "description",
            "source",
            "license",
            "access",
            "bids",
            "modalities",
            "optional_per_subject_modalities",
            "expected",
            "intended_use",
            "prohibited_use",
            "acknowledgement",
            "ethics_approval",
        }
        _keys(payload, required=required, optional=set(), location="dataset registry")
        schema_version = _string(payload["schema_version"], "schema_version")
        if schema_version != _REGISTRY_SCHEMA_VERSION:
            raise DatasetRegistryError(
                f"schema_version must be {_REGISTRY_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        modalities = _strings(payload["modalities"], "modalities")
        optional = _strings(
            payload["optional_per_subject_modalities"], "optional_per_subject_modalities"
        )
        if not set(optional).issubset(modalities):
            raise DatasetRegistryError(
                "optional_per_subject_modalities must be a subset of modalities"
            )
        return cls(
            schema_version=schema_version,
            dataset_id=_string(payload["dataset_id"], "dataset_id"),
            dataset_version=_string(payload["dataset_version"], "dataset_version"),
            name=_string(payload["name"], "name"),
            description=_string(payload["description"], "description"),
            source=DatasetSource.from_mapping(payload["source"]),
            license=DatasetLicense.from_mapping(payload["license"]),
            access=DatasetAccess.from_mapping(payload["access"]),
            bids=BidsContract.from_mapping(payload["bids"]),
            modalities=modalities,
            optional_per_subject_modalities=optional,
            expected=ExpectedDataset.from_mapping(payload["expected"]),
            intended_use=_string(payload["intended_use"], "intended_use"),
            prohibited_use=_string(payload["prohibited_use"], "prohibited_use"),
            acknowledgement=_string(payload["acknowledgement"], "acknowledgement"),
            ethics_approval=_string(payload["ethics_approval"], "ethics_approval"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset_registry(path: Path) -> DatasetRegistryEntry:
    """Load a strict YAML dataset-registry entry."""

    if not path.is_file():
        raise DatasetRegistryError(f"dataset registry does not exist: {path.name}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetRegistryError(f"could not read dataset registry {path.name}: {exc}") from exc
    return DatasetRegistryEntry.from_mapping(payload)


def resolve_dataset_root(
    registry: DatasetRegistryEntry,
    *,
    explicit_root: Path | None,
    environment: dict[str, str] | None = None,
) -> tuple[Path, Literal["explicit", "environment"]]:
    """Resolve a dataset root without including it in reports or exceptions."""

    if explicit_root is not None:
        return explicit_root.expanduser(), "explicit"
    values = os.environ if environment is None else environment
    base = values.get(registry.access.environment_variable)
    if not base:
        raise DatasetRegistryError(
            f"set {registry.access.environment_variable} or pass --dataset-root"
        )
    return Path(base).expanduser() / registry.access.relative_path, "environment"


def _path_fingerprint(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:16]


def _read_participants(path: Path) -> tuple[list[str], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = list(reader.fieldnames or [])
            if "participant_id" not in fields:
                raise DatasetRegistryError("participants.tsv is missing participant_id")
            participants: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                participant = (row.get("participant_id") or "").strip()
                if not _SUBJECT_PATTERN.fullmatch(participant):
                    raise DatasetRegistryError(
                        f"participants.tsv row {row_number} has an invalid participant_id"
                    )
                participants.append(participant)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetRegistryError("participants.tsv is unreadable") from exc
    if len(participants) != len(set(participants)):
        raise DatasetRegistryError("participants.tsv contains duplicate participant IDs")
    return participants, fields


def _named_payloads(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    matches: list[Path] = []
    for candidate in directory.rglob("*"):
        if candidate.name.endswith(suffixes):
            matches.append(candidate)
    return sorted(matches)


@dataclass(frozen=True)
class SubjectInventory:
    participant_id: str
    anat_present: bool
    t1w_count: int
    eeg_present: bool
    func_present: bool
    dwi_present: bool


@dataclass(frozen=True)
class DatasetValidation:
    status: Literal["pass", "fail"]
    dataset_id: str
    dataset_version: str
    registry_schema_version: str
    root_source: Literal["explicit", "environment"]
    root_fingerprint: str
    participant_count: int
    expected_participant_count: int
    inventory: tuple[SubjectInventory, ...]
    checks: tuple[dict[str, object], ...]
    warnings: tuple[str, ...]
    scientific_data_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_local_dataset(
    registry: DatasetRegistryEntry,
    dataset_root: Path,
    *,
    root_source: Literal["explicit", "environment"],
) -> DatasetValidation:
    """Validate identity and layout of a local dataset without full BIDS validation."""

    checks: list[dict[str, object]] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})

    if not dataset_root.is_dir():
        raise DatasetRegistryError("resolved dataset root is missing or is not a directory")

    for file_name in registry.expected.top_level_files:
        check(f"top_level:{file_name}", (dataset_root / file_name).is_file(), "required file")

    description_path = dataset_root / "dataset_description.json"
    try:
        description = json.loads(description_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetRegistryError("dataset_description.json is unreadable") from exc

    expected_description = {
        "Name": registry.name,
        "BIDSVersion": registry.bids.version,
        "DatasetType": registry.bids.dataset_type,
        "License": registry.license.identifier,
        "DatasetDOI": f"doi:{registry.source.doi}",
    }
    for field, expected in expected_description.items():
        observed = description.get(field)
        check(f"dataset_description:{field}", observed == expected, f"expected {expected!r}")

    participants_path = dataset_root / "participants.tsv"
    participants, participant_fields = _read_participants(participants_path)
    check(
        "participant_count",
        len(participants) == registry.expected.participant_count,
        f"observed {len(participants)}, expected {registry.expected.participant_count}",
    )

    directories = sorted(
        item.name
        for item in dataset_root.iterdir()
        if item.is_dir() and _SUBJECT_PATTERN.fullmatch(item.name)
    )
    participant_set = set(participants)
    directory_set = set(directories)
    check(
        "participant_directories",
        participant_set == directory_set,
        "participants.tsv and subject directories must match exactly",
    )

    inventory: list[SubjectInventory] = []
    for participant in participants:
        subject_root = dataset_root / participant
        t1w = _named_payloads(subject_root / "anat", ("_T1w.nii", "_T1w.nii.gz"))
        inventory.append(
            SubjectInventory(
                participant_id=participant,
                anat_present=(subject_root / "anat").is_dir(),
                t1w_count=len(t1w),
                eeg_present=(subject_root / "eeg").is_dir(),
                func_present=(subject_root / "func").is_dir(),
                dwi_present=(subject_root / "dwi").is_dir(),
            )
        )

    if registry.expected.all_participants_require_t1w:
        missing_t1w = [item.participant_id for item in inventory if item.t1w_count == 0]
        check(
            "all_participants_have_t1w_layout",
            not missing_t1w,
            f"subjects without a T1w filename: {len(missing_t1w)}",
        )

    modality_counts = {
        "anat": sum(item.anat_present for item in inventory),
        "eeg": sum(item.eeg_present for item in inventory),
        "func": sum(item.func_present for item in inventory),
        "dwi": sum(item.dwi_present for item in inventory),
    }
    for modality in registry.modalities:
        if modality_counts.get(modality, 0) == 0:
            warnings.append(f"no subject directory was found for declared modality {modality}")

    if "age" in participant_fields or "sex" in participant_fields:
        warnings.append(
            "participants.tsv contains participant attributes; reports intentionally omit "
            "their values"
        )

    status: Literal["pass", "fail"] = (
        "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    )
    return DatasetValidation(
        status=status,
        dataset_id=registry.dataset_id,
        dataset_version=registry.dataset_version,
        registry_schema_version=registry.schema_version,
        root_source=root_source,
        root_fingerprint=_path_fingerprint(dataset_root),
        participant_count=len(participants),
        expected_participant_count=registry.expected.participant_count,
        inventory=tuple(inventory),
        checks=tuple(checks),
        warnings=tuple(warnings),
        scientific_data_ready=False,
    )


def write_dataset_validation_report(
    *,
    registry: DatasetRegistryEntry,
    validation: DatasetValidation,
    dataset_root: Path,
    registry_path: Path,
    output_dir: Path,
) -> None:
    """Write non-sensitive machine and human-readable registration reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dataset_validation.json", validation.to_dict())
    write_json(output_dir / "registry_snapshot.json", registry.to_dict())

    inventory_path = output_dir / "subject_modality_inventory.tsv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "participant_id",
                "anat_present",
                "t1w_count",
                "eeg_present",
                "func_present",
                "dwi_present",
            ]
        )
        for item in validation.inventory:
            writer.writerow(
                [
                    item.participant_id,
                    str(item.anat_present).lower(),
                    item.t1w_count,
                    str(item.eeg_present).lower(),
                    str(item.func_present).lower(),
                    str(item.dwi_present).lower(),
                ]
            )

    check_lines = [
        (
            f"- **{str(check_item['status']).upper()}** — "
            f"`{check_item['name']}`: {check_item['detail']}"
        )
        for check_item in validation.checks
    ]
    warning_lines = [f"- {warning}" for warning in validation.warnings] or ["- None"]
    summary = "\n".join(
        [
            f"# Dataset registration validation — {validation.dataset_id}",
            "",
            f"- Version: `{validation.dataset_version}`",
            f"- Status: **{validation.status.upper()}**",
            f"- Participants observed: **{validation.participant_count}**",
            f"- Dataset-root source: `{validation.root_source}`",
            f"- Dataset-root fingerprint: `{validation.root_fingerprint}`",
            "- Full BIDS and payload validation: **not performed in Phase 03**",
            "- Scientific-data readiness: **false until Phase 04**",
            "",
            "## Checks",
            "",
            *check_lines,
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
            "## Privacy note",
            "",
            "This report records no absolute local path and no participant attributes.",
            "",
        ]
    )
    (output_dir / "dataset_validation.md").write_text(summary, encoding="utf-8")

    checksum_inputs = [
        registry_path,
        dataset_root / "dataset_description.json",
        dataset_root / "participants.tsv",
    ]
    checksum_inputs.extend(
        dataset_root / file_name
        for file_name in registry.expected.top_level_files
        if (dataset_root / file_name).is_file()
    )
    unique_inputs = sorted({item.resolve() for item in checksum_inputs if item.is_file()})
    source_manifest = output_dir / "source_metadata_checksums.sha256"
    lines: list[str] = []
    for source_path in unique_inputs:
        role = (
            "registry"
            if source_path == registry_path.resolve()
            else f"dataset_metadata/{source_path.name}"
        )
        lines.append(f"{sha256_file(source_path)}  {role}")
    source_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    generated = [item for item in output_dir.iterdir() if item.is_file()]
    write_checksum_manifest(output_dir, generated, output_dir / "checksums.sha256")
