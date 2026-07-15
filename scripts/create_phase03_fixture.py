#!/usr/bin/env python3
"""Create a metadata-only ds004024 fixture for tests and CI.

The fixture contains no neuroimaging or electrophysiology payloads. Empty files only
exercise dataset identity and directory-layout validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

DESCRIPTION = {
    "Name": (
        "TMS-EEG-MRI-fMRI-DWI data on paired associative stimulation and connectivity "
        "(Shirley Ryan AbilityLab, Chicago, IL)"
    ),
    "BIDSVersion": "1.6.0",
    "DatasetType": "raw",
    "Authors": [
        "Julio Cesar Hernandez Pavon",
        "Nils Schneider Garces",
        "John Patrick Begnoche",
        "Lee Miller",
        "Tommi Raij",
    ],
    "License": "CC0",
    "EthicsApprovals": ["STU00204239"],
    "DatasetDOI": "doi:10.18112/openneuro.ds004024.v1.0.1",
}


def create_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "dataset_description.json").write_text(
        json.dumps(DESCRIPTION, indent=2) + "\n", encoding="utf-8"
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\thand\n"
        + "".join(f"{participant}\tn/a\tn/a\tn/a\n" for participant in PARTICIPANTS),
        encoding="utf-8",
    )
    (root / "participants.json").write_text("{}\n", encoding="utf-8")
    (root / "README").write_text(
        "Metadata-only CI fixture; not the OpenNeuro dataset.\n", encoding="utf-8"
    )
    (root / "CHANGES").write_text("Synthetic fixture.\n", encoding="utf-8")
    for index, participant in enumerate(PARTICIPANTS):
        subject = root / participant
        (subject / "anat").mkdir(parents=True)
        (subject / "eeg").mkdir()
        (subject / "anat" / f"{participant}_T1w.nii.gz").touch()
        (subject / "eeg" / f"{participant}_task-fixture_eeg.set").touch()
        if index % 2 == 0:
            (subject / "func").mkdir()
        if index % 3 == 0:
            (subject / "dwi").mkdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.output)
    print(args.output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
