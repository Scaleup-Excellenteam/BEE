from pathlib import Path
from zipfile import ZipFile

import pytest

from src.corpus.archive import extract_archive


def _write_zip(zip_path: Path, entries: dict[str, str]) -> None:
    with ZipFile(zip_path, "w") as zip_file:
        for name, content in entries.items():
            zip_file.writestr(name, content)


def test_extracts_files_and_nested_directories(tmp_path):
    archive_path = tmp_path / "small.zip"
    _write_zip(
        archive_path,
        {
            "a.txt": "first\n",
            "nested/b.txt": "second\n",
        },
    )
    destination = tmp_path / "out"

    returned = extract_archive(str(archive_path), str(destination))

    assert Path(returned) == destination
    assert (destination / "a.txt").read_text() == "first\n"
    assert (destination / "nested" / "b.txt").read_text() == "second\n"


def test_creates_destination_when_missing(tmp_path):
    archive_path = tmp_path / "small.zip"
    _write_zip(archive_path, {"a.txt": "x"})
    destination = tmp_path / "does" / "not" / "exist"

    extract_archive(str(archive_path), str(destination))

    assert destination.is_dir()


def test_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_archive(str(tmp_path / "nope.zip"), str(tmp_path / "out"))


def test_rejects_path_traversal_entry(tmp_path):
    archive_path = tmp_path / "evil.zip"
    _write_zip(archive_path, {"../escaped.txt": "pwned"})
    destination = tmp_path / "out"

    with pytest.raises(ValueError):
        extract_archive(str(archive_path), str(destination))

    assert not (tmp_path / "escaped.txt").exists()
