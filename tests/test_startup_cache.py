import sqlite3
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

import autocomplete
import src.corpus.initialization as initialization
from src.corpus.index import CorpusIndex
from src.corpus.persistence import LEXICAL_BUILD_VERSION


def _write_archive(archive_path, entries):
    with ZipFile(archive_path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _paths(tmp_path):
    return (
        tmp_path / "Archive.zip",
        tmp_path / "extracted",
        tmp_path / "corpus.sqlite3",
    )


def _load_or_initialize(archive, extraction, snapshot):
    return initialization.load_or_initialize_corpus(
        str(archive),
        str(extraction),
        str(snapshot),
    )


def test_first_startup_cold_builds_and_creates_snapshot(tmp_path):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"docs/a.txt": "Computer Science\n"})

    index = _load_or_initialize(archive, extraction, snapshot)

    assert isinstance(index, CorpusIndex)
    assert [record.original_sentence for record in index.records] == [
        "Computer Science"
    ]
    assert snapshot.is_file()
    assert (extraction / "docs" / "a.txt").is_file()


def test_second_startup_loads_snapshot_and_bypasses_cold_path(
    tmp_path,
    monkeypatch,
):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"a.txt": "Hello World\n"})
    expected = _load_or_initialize(archive, extraction, snapshot)

    def fail(*_args, **_kwargs):
        raise AssertionError("cold corpus work must be bypassed on cache hit")

    monkeypatch.setattr(initialization, "extract_archive", fail)
    monkeypatch.setattr(initialization, "load_corpus", fail)
    monkeypatch.setattr(CorpusIndex, "_build", fail)

    loaded = _load_or_initialize(archive, extraction, snapshot)

    assert loaded.records == expected.records
    assert loaded.get_candidates("hello") == expected.get_candidates("hello")


def test_changed_archive_rebuilds_and_replaces_snapshot(tmp_path, monkeypatch):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"old.txt": "Old sentence\n"})
    _load_or_initialize(archive, extraction, snapshot)

    _write_archive(archive, {"new.txt": "New sentence\n"})
    cold_build = Mock(wraps=initialization.initialize_corpus)
    monkeypatch.setattr(initialization, "initialize_corpus", cold_build)

    rebuilt = _load_or_initialize(archive, extraction, snapshot)

    assert cold_build.call_count == 1
    assert [record.original_sentence for record in rebuilt.records] == [
        "New sentence"
    ]
    assert not (extraction / "old.txt").exists()

    cold_build.reset_mock()
    loaded = _load_or_initialize(archive, extraction, snapshot)
    assert cold_build.call_count == 0
    assert loaded.records == rebuilt.records


def test_corrupt_snapshot_falls_back_to_rebuild(tmp_path, monkeypatch):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"a.txt": "Usable corpus\n"})
    snapshot.write_bytes(b"not a SQLite database")
    cold_build = Mock(wraps=initialization.initialize_corpus)
    monkeypatch.setattr(initialization, "initialize_corpus", cold_build)

    index = _load_or_initialize(archive, extraction, snapshot)

    assert cold_build.call_count == 1
    assert index.records[0].original_sentence == "Usable corpus"
    assert snapshot.read_bytes().startswith(b"SQLite format 3")


def test_incompatible_snapshot_falls_back_to_rebuild(tmp_path, monkeypatch):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"a.txt": "Compatible corpus\n"})
    _load_or_initialize(archive, extraction, snapshot)

    with sqlite3.connect(snapshot) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = ?",
            (LEXICAL_BUILD_VERSION + "-old", "lexical_build_version"),
        )

    cold_build = Mock(wraps=initialization.initialize_corpus)
    monkeypatch.setattr(initialization, "initialize_corpus", cold_build)

    index = _load_or_initialize(archive, extraction, snapshot)

    assert cold_build.call_count == 1
    assert index.records[0].original_sentence == "Compatible corpus"


def test_loaded_and_rebuilt_paths_have_equivalent_autocomplete_results(
    tmp_path,
    monkeypatch,
):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(
        archive,
        {
            "docs/a.txt": "To be or not to be, that is the question.\n",
            "docs/b.txt": "Unrelated sentence\n",
        },
    )
    rebuilt = _load_or_initialize(archive, extraction, snapshot)
    loaded = _load_or_initialize(archive, extraction, snapshot)

    monkeypatch.setattr(autocomplete, "_corpus_index", rebuilt)
    rebuilt_results = {
        query: autocomplete.get_best_k_completions(query)
        for query in ("to be", "2o be", "or nt", "or knot")
    }
    monkeypatch.setattr(autocomplete, "_corpus_index", loaded)
    loaded_results = {
        query: autocomplete.get_best_k_completions(query)
        for query in ("to be", "2o be", "or nt", "or knot")
    }

    assert loaded_results == rebuilt_results


def test_cold_build_removes_stale_extracted_files_and_preserves_archive(
    tmp_path,
):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"current.txt": "Current sentence\n"})
    original_archive = archive.read_bytes()
    extraction.mkdir()
    (extraction / "stale.txt").write_text("Stale sentence\n", encoding="utf-8")

    index = _load_or_initialize(archive, extraction, snapshot)

    assert [record.source_text for record in index.records] == ["current.txt"]
    assert not (extraction / "stale.txt").exists()
    assert archive.read_bytes() == original_archive


def test_cache_save_failure_does_not_discard_cold_built_index(
    tmp_path,
    monkeypatch,
):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"a.txt": "Still starts\n"})
    monkeypatch.setattr(
        initialization,
        "save_corpus_index",
        Mock(side_effect=OSError("cache is unwritable")),
    )

    index = _load_or_initialize(archive, extraction, snapshot)

    assert index.records[0].original_sentence == "Still starts"


def test_cold_build_error_is_not_hidden(tmp_path, monkeypatch):
    archive, extraction, snapshot = _paths(tmp_path)
    _write_archive(archive, {"a.txt": "line\n"})
    monkeypatch.setattr(
        initialization,
        "initialize_corpus",
        Mock(side_effect=RuntimeError("cold build failed")),
    )

    with pytest.raises(RuntimeError, match="cold build failed"):
        _load_or_initialize(archive, extraction, snapshot)
