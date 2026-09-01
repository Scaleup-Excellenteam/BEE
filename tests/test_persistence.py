import hashlib
import sqlite3
import struct
from array import array

import pytest

import autocomplete
from src.corpus.index import CorpusIndex
from src.corpus.persistence import (
    LEXICAL_BUILD_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotCompatibilityError,
    SnapshotCorruptionError,
    calculate_archive_sha256,
    load_corpus_index,
    save_corpus_index,
)
from src.models import SentenceRecord


ARCHIVE_SHA256 = hashlib.sha256(b"test archive").hexdigest()


def _records() -> list[SentenceRecord]:
    return [
        SentenceRecord(
            original_sentence="Computer Science!",
            normalized_sentence="computer science",
            source_text="docs/computing.txt",
            offset=7,
        ),
        SentenceRecord(
            original_sentence="Hello, World",
            normalized_sentence="hello world",
            source_text="docs/greetings.txt",
            offset=13,
        ),
        SentenceRecord(
            original_sentence="A short line",
            normalized_sentence="a short line",
            source_text="root.txt",
            offset=2,
        ),
    ]


def _round_trip(tmp_path, index: CorpusIndex) -> CorpusIndex:
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(index, snapshot, archive_sha256=ARCHIVE_SHA256)
    return load_corpus_index(
        snapshot,
        expected_archive_sha256=ARCHIVE_SHA256,
    )


def test_calculates_archive_sha256(tmp_path):
    archive = tmp_path / "small.zip"
    archive.write_bytes(b"archive bytes")

    assert calculate_archive_sha256(archive) == hashlib.sha256(
        b"archive bytes"
    ).hexdigest()


def test_save_load_round_trip_preserves_records_postings_and_stats(tmp_path):
    fresh = CorpusIndex(_records())
    restored = _round_trip(tmp_path, fresh)

    assert isinstance(restored, CorpusIndex)
    assert restored.records == fresh.records
    assert restored.records is not fresh.records
    assert [
        (
            record.original_sentence,
            record.normalized_sentence,
            record.source_text,
            record.offset,
        )
        for record in restored.records
    ] == [
        (
            record.original_sentence,
            record.normalized_sentence,
            record.source_text,
            record.offset,
        )
        for record in fresh.records
    ]
    assert {
        trigram: list(sentence_ids)
        for trigram, sentence_ids in restored._postings.items()
    } == {
        trigram: list(sentence_ids)
        for trigram, sentence_ids in fresh._postings.items()
    }
    assert all(
        isinstance(sentence_ids, array)
        and sentence_ids.typecode == "I"
        for sentence_ids in restored._postings.values()
    )
    assert restored._total_postings == fresh._total_postings
    assert restored._build_seconds == fresh._build_seconds


def test_load_does_not_run_corpus_index_build(tmp_path, monkeypatch):
    fresh = CorpusIndex(_records())
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(fresh, snapshot, archive_sha256=ARCHIVE_SHA256)

    def fail_if_called(_self):
        raise AssertionError("CorpusIndex._build() must not run during restore")

    monkeypatch.setattr(CorpusIndex, "_build", fail_if_called)

    restored = load_corpus_index(
        snapshot,
        expected_archive_sha256=ARCHIVE_SHA256,
    )

    assert restored.records == fresh.records


def test_fresh_and_restored_get_candidates_are_equivalent(tmp_path):
    fresh = CorpusIndex(_records())
    restored = _round_trip(tmp_path, fresh)

    for query in ("computer sci", "puter sci", "helo", "ab", ""):
        assert restored.get_candidates(query) == fresh.get_candidates(query)


def test_fresh_and_restored_autocomplete_are_equivalent_for_all_match_types(
    tmp_path,
    monkeypatch,
):
    records = [
        SentenceRecord(
            original_sentence="To be or not to be, that is the question.",
            normalized_sentence="to be or not to be that is the question",
            source_text="hamlet.txt",
            offset=42,
        ),
        SentenceRecord(
            original_sentence="axc",
            normalized_sentence="axc",
            source_text="zero.txt",
            offset=3,
        ),
        SentenceRecord(
            original_sentence="b",
            normalized_sentence="b",
            source_text="negative.txt",
            offset=9,
        ),
    ]
    fresh = CorpusIndex(records)
    restored = _round_trip(tmp_path, fresh)
    queries = ("to be", "2o be", "or nt", "or knot", "abc", "a")

    monkeypatch.setattr(autocomplete, "_corpus_index", fresh)
    fresh_results = {
        query: autocomplete.get_best_k_completions(query)
        for query in queries
    }
    monkeypatch.setattr(autocomplete, "_corpus_index", restored)
    restored_results = {
        query: autocomplete.get_best_k_completions(query)
        for query in queries
    }

    assert restored_results == fresh_results
    assert fresh_results["to be"][0].score == 10
    assert fresh_results["2o be"][0].score == 3
    assert fresh_results["or nt"][0].score == 8
    assert fresh_results["or knot"][0].score == 8
    assert fresh_results["abc"][0].score == 0
    assert next(
        result.score
        for result in fresh_results["a"]
        if result.completed_sentence == "b"
    ) == -5


def test_corrupt_snapshot_is_rejected(tmp_path):
    snapshot = tmp_path / "corrupt.sqlite"
    snapshot.write_bytes(b"not a SQLite database")

    with pytest.raises(SnapshotCorruptionError, match="Invalid corpus snapshot"):
        load_corpus_index(
            snapshot,
            expected_archive_sha256=ARCHIVE_SHA256,
        )


@pytest.mark.parametrize(
    ("metadata_key", "replacement", "message"),
    [
        (
            "schema_version",
            str(SNAPSHOT_SCHEMA_VERSION + 1),
            "schema version mismatch",
        ),
        (
            "lexical_build_version",
            LEXICAL_BUILD_VERSION + "-changed",
            "lexical-build version mismatch",
        ),
    ],
)
def test_version_mismatch_is_rejected(
    tmp_path,
    metadata_key,
    replacement,
    message,
):
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(
        CorpusIndex(_records()),
        snapshot,
        archive_sha256=ARCHIVE_SHA256,
    )
    with sqlite3.connect(snapshot) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = ?",
            (replacement, metadata_key),
        )

    with pytest.raises(SnapshotCompatibilityError, match=message):
        load_corpus_index(
            snapshot,
            expected_archive_sha256=ARCHIVE_SHA256,
        )


def test_archive_fingerprint_mismatch_is_rejected(tmp_path):
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(
        CorpusIndex(_records()),
        snapshot,
        archive_sha256=ARCHIVE_SHA256,
    )

    with pytest.raises(SnapshotCompatibilityError, match="fingerprint"):
        load_corpus_index(
            snapshot,
            expected_archive_sha256=hashlib.sha256(b"changed").hexdigest(),
        )


def test_invalid_posting_id_is_rejected(tmp_path):
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(
        CorpusIndex(_records()),
        snapshot,
        archive_sha256=ARCHIVE_SHA256,
    )
    with sqlite3.connect(snapshot) as connection:
        connection.execute(
            """
            UPDATE postings
            SET sentence_ids = ?, posting_count = 1
            WHERE trigram = (SELECT trigram FROM postings LIMIT 1)
            """,
            (sqlite3.Binary(struct.pack("<I", 999_999)),),
        )

    with pytest.raises(SnapshotCorruptionError, match="out-of-range"):
        load_corpus_index(
            snapshot,
            expected_archive_sha256=ARCHIVE_SHA256,
        )


@pytest.mark.parametrize(
    ("sentence_ids", "message"),
    [
        ((1, 0), "not sorted"),
        ((0, 0), "duplicate sentence ID"),
    ],
)
def test_unsorted_or_duplicate_posting_ids_are_rejected(
    tmp_path,
    sentence_ids,
    message,
):
    snapshot = tmp_path / "corpus.sqlite"
    save_corpus_index(
        CorpusIndex(_records()),
        snapshot,
        archive_sha256=ARCHIVE_SHA256,
    )
    with sqlite3.connect(snapshot) as connection:
        connection.execute(
            """
            UPDATE postings
            SET sentence_ids = ?, posting_count = 2
            WHERE trigram = (SELECT trigram FROM postings LIMIT 1)
            """,
            (sqlite3.Binary(struct.pack("<II", *sentence_ids)),),
        )

    with pytest.raises(SnapshotCorruptionError, match=message):
        load_corpus_index(
            snapshot,
            expected_archive_sha256=ARCHIVE_SHA256,
        )
