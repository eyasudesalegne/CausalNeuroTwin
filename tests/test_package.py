from __future__ import annotations

import causalneurotwin


def test_package_import_and_version() -> None:
    assert causalneurotwin.__version__ == "0.1.0a1"
