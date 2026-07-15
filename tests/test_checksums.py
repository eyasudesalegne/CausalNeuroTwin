from __future__ import annotations

from pathlib import Path

from causalneurotwin.checksums import sha256_file, write_checksum_manifest


def test_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_manifest_is_sorted_and_excludes_itself(tmp_path: Path) -> None:
    second = tmp_path / "b.txt"
    first = tmp_path / "a.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    destination = tmp_path / "checksums.sha256"
    write_checksum_manifest(tmp_path, [second, destination, first], destination)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("  a.txt")
    assert lines[1].endswith("  b.txt")
    assert all("checksums.sha256" not in line for line in lines)
