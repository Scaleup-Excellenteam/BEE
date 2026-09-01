"""Versioned SQLite snapshots for a fully built :class:`CorpusIndex`.

SQLite is used only as a safe, transactional persistence container.  Search
continues to use BEE's in-memory ``CorpusIndex`` and its existing trigram
posting arrays.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import sys
import tempfile
from array import array
from pathlib import Path

from src.corpus.index import CorpusIndex, POSTING_TYPE_CODE
from src.models import SentenceRecord


SNAPSHOT_SCHEMA_VERSION = 1
LEXICAL_BUILD_VERSION = "bee-lexical-trigram-v1"
_POSTING_WIDTH_BYTES = 4
_POSTING_BYTE_ORDER = "little"
_HASH_CHUNK_BYTES = 1024 * 1024


class CorpusSnapshotError(RuntimeError):
    """Base error for an unusable corpus snapshot."""


class SnapshotCompatibilityError(CorpusSnapshotError):
    """The snapshot is valid but does not match the requested corpus/runtime."""


class SnapshotCorruptionError(CorpusSnapshotError):
    """The snapshot is malformed, incomplete, or internally inconsistent."""


def calculate_archive_sha256(archive_path: str | Path) -> str:
    """Return the lowercase SHA-256 content digest of an archive."""
    digest = hashlib.sha256()
    with Path(archive_path).open("rb") as archive:
        while chunk := archive.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def save_corpus_index(
    index: CorpusIndex,
    snapshot_path: str | Path,
    *,
    archive_sha256: str,
) -> None:
    """Atomically save a ready ``CorpusIndex`` to a SQLite snapshot."""
    archive_digest = _validated_sha256(archive_sha256)
    records = _validated_records(index.records)
    postings, actual_posting_count = _validated_postings(
        index._postings,
        len(records),
    )

    if index._total_postings != actual_posting_count:
        raise CorpusSnapshotError(
            "CorpusIndex total-posting count does not match its postings"
        )

    build_seconds = float(index._build_seconds)
    if not math.isfinite(build_seconds) or build_seconds < 0:
        raise CorpusSnapshotError("CorpusIndex build time is invalid")

    destination = Path(snapshot_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                f"PRAGMA user_version = {SNAPSHOT_SCHEMA_VERSION}"
            )
            _create_schema(connection)

            metadata = {
                "schema_version": str(SNAPSHOT_SCHEMA_VERSION),
                "lexical_build_version": LEXICAL_BUILD_VERSION,
                "archive_sha256": archive_digest,
                "record_count": str(len(records)),
                "trigram_count": str(len(postings)),
                "posting_count": str(actual_posting_count),
                "build_seconds": repr(build_seconds),
                "posting_width_bytes": str(_POSTING_WIDTH_BYTES),
                "posting_byte_order": _POSTING_BYTE_ORDER,
                "snapshot_complete": "1",
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.executemany(
                """
                INSERT INTO records(
                    sentence_id,
                    original_sentence,
                    normalized_sentence,
                    source_text,
                    offset
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        sentence_id,
                        record.original_sentence,
                        record.normalized_sentence,
                        record.source_text,
                        record.offset,
                    )
                    for sentence_id, record in enumerate(records)
                ),
            )
            connection.executemany(
                """
                INSERT INTO postings(trigram, sentence_ids, posting_count)
                VALUES (?, ?, ?)
                """,
                (
                    (
                        trigram,
                        sqlite3.Binary(_encode_posting_ids(sentence_ids)),
                        len(sentence_ids),
                    )
                    for trigram, sentence_ids in postings.items()
                ),
            )
            connection.commit()
        finally:
            connection.close()

        with temporary_path.open("rb") as snapshot:
            os.fsync(snapshot.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except sqlite3.Error as error:
        raise CorpusSnapshotError(
            f"Could not write corpus snapshot: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def load_corpus_index(
    snapshot_path: str | Path,
    *,
    expected_archive_sha256: str,
) -> CorpusIndex:
    """Load and fully validate a ready ``CorpusIndex`` without rebuilding."""
    expected_digest = _validated_sha256(expected_archive_sha256)
    path = Path(snapshot_path)
    if not path.is_file():
        raise CorpusSnapshotError(f"Corpus snapshot not found: {path}")

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        metadata = _read_metadata(connection)
        _validate_compatibility(connection, metadata, expected_digest)

        record_count = _metadata_nonnegative_int(metadata, "record_count")
        trigram_count = _metadata_nonnegative_int(metadata, "trigram_count")
        posting_count = _metadata_nonnegative_int(metadata, "posting_count")
        build_seconds = _metadata_build_seconds(metadata)

        records = _read_records(connection, record_count)
        postings, actual_posting_count = _read_postings(
            connection,
            record_count,
            trigram_count,
        )

        if actual_posting_count != posting_count:
            raise SnapshotCorruptionError(
                "Snapshot posting count does not match its metadata"
            )

        restored = CorpusIndex.__new__(CorpusIndex)
        restored.records = records
        restored._postings = postings
        restored._total_postings = posting_count
        restored._build_seconds = build_seconds
        return restored
    except CorpusSnapshotError:
        raise
    except (sqlite3.Error, OverflowError, TypeError, ValueError) as error:
        raise SnapshotCorruptionError(
            f"Invalid corpus snapshot {path}: {error}"
        ) from error
    finally:
        if connection is not None:
            connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        );

        CREATE TABLE records (
            sentence_id INTEGER PRIMARY KEY NOT NULL,
            original_sentence TEXT NOT NULL,
            normalized_sentence TEXT NOT NULL,
            source_text TEXT NOT NULL,
            offset INTEGER NOT NULL
        );

        CREATE TABLE postings (
            trigram TEXT PRIMARY KEY NOT NULL,
            sentence_ids BLOB NOT NULL,
            posting_count INTEGER NOT NULL CHECK (posting_count >= 0)
        );
        """
    )


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    metadata: dict[str, str] = {}
    for key, value in rows:
        if not isinstance(key, str) or not isinstance(value, str):
            raise SnapshotCorruptionError("Snapshot metadata has invalid types")
        if key in metadata:
            raise SnapshotCorruptionError(f"Duplicate snapshot metadata key: {key}")
        metadata[key] = value
    return metadata


def _validate_compatibility(
    connection: sqlite3.Connection,
    metadata: dict[str, str],
    expected_archive_sha256: str,
) -> None:
    required_keys = {
        "schema_version",
        "lexical_build_version",
        "archive_sha256",
        "record_count",
        "trigram_count",
        "posting_count",
        "build_seconds",
        "posting_width_bytes",
        "posting_byte_order",
        "snapshot_complete",
    }
    missing = sorted(required_keys - metadata.keys())
    if missing:
        raise SnapshotCorruptionError(
            f"Snapshot metadata is incomplete; missing: {', '.join(missing)}"
        )

    sqlite_schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    try:
        metadata_schema_version = int(metadata["schema_version"])
    except ValueError as error:
        raise SnapshotCorruptionError("Snapshot schema version is invalid") from error

    if (
        sqlite_schema_version != SNAPSHOT_SCHEMA_VERSION
        or metadata_schema_version != SNAPSHOT_SCHEMA_VERSION
    ):
        raise SnapshotCompatibilityError(
            "Snapshot schema version mismatch: "
            f"expected {SNAPSHOT_SCHEMA_VERSION}, got "
            f"SQLite={sqlite_schema_version}, metadata={metadata_schema_version}"
        )

    lexical_version = metadata["lexical_build_version"]
    if lexical_version != LEXICAL_BUILD_VERSION:
        raise SnapshotCompatibilityError(
            "Snapshot lexical-build version mismatch: "
            f"expected {LEXICAL_BUILD_VERSION!r}, got {lexical_version!r}"
        )

    snapshot_digest = _snapshot_sha256(metadata["archive_sha256"])
    if snapshot_digest != expected_archive_sha256:
        raise SnapshotCompatibilityError(
            "Snapshot archive fingerprint does not match the requested archive"
        )

    if metadata["posting_width_bytes"] != str(_POSTING_WIDTH_BYTES):
        raise SnapshotCompatibilityError("Snapshot posting width is incompatible")
    if metadata["posting_byte_order"] != _POSTING_BYTE_ORDER:
        raise SnapshotCompatibilityError(
            "Snapshot posting byte order is incompatible"
        )
    if metadata["snapshot_complete"] != "1":
        raise SnapshotCorruptionError("Snapshot was not completed")


def _read_records(
    connection: sqlite3.Connection,
    expected_count: int,
) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    rows = connection.execute(
        """
        SELECT sentence_id, original_sentence, normalized_sentence,
               source_text, offset
        FROM records
        ORDER BY sentence_id
        """
    )
    for expected_id, row in enumerate(rows):
        sentence_id, original, normalized, source, offset = row
        if sentence_id != expected_id:
            raise SnapshotCorruptionError(
                "Snapshot record IDs are not contiguous and ordered"
            )
        if not all(isinstance(value, str) for value in (original, normalized, source)):
            raise SnapshotCorruptionError("Snapshot record contains non-text fields")
        if not isinstance(offset, int):
            raise SnapshotCorruptionError("Snapshot record offset is not an integer")
        records.append(
            SentenceRecord(
                original_sentence=original,
                normalized_sentence=normalized,
                source_text=source,
                offset=offset,
            )
        )

    if len(records) != expected_count:
        raise SnapshotCorruptionError(
            "Snapshot record count does not match its metadata"
        )
    return records


def _read_postings(
    connection: sqlite3.Connection,
    record_count: int,
    expected_trigram_count: int,
) -> tuple[dict[str, array], int]:
    postings: dict[str, array] = {}
    total_postings = 0
    rows = connection.execute(
        "SELECT trigram, sentence_ids, posting_count FROM postings"
    )

    for trigram, encoded_ids, declared_count in rows:
        if not isinstance(trigram, str):
            raise SnapshotCorruptionError("Snapshot trigram is not text")
        if trigram in postings:
            raise SnapshotCorruptionError(f"Duplicate trigram in snapshot: {trigram!r}")
        if not isinstance(encoded_ids, bytes):
            raise SnapshotCorruptionError(
                f"Posting payload for {trigram!r} is not binary"
            )
        if not isinstance(declared_count, int) or declared_count < 0:
            raise SnapshotCorruptionError(
                f"Posting count for {trigram!r} is invalid"
            )

        sentence_ids = _decode_posting_ids(encoded_ids)
        if len(sentence_ids) != declared_count:
            raise SnapshotCorruptionError(
                f"Posting count for {trigram!r} does not match its payload"
            )
        _validate_posting_ids(trigram, sentence_ids, record_count)
        postings[trigram] = sentence_ids
        total_postings += len(sentence_ids)

    if len(postings) != expected_trigram_count:
        raise SnapshotCorruptionError(
            "Snapshot trigram count does not match its metadata"
        )
    return postings, total_postings


def _validated_records(records: list[SentenceRecord]) -> list[SentenceRecord]:
    if not isinstance(records, list):
        raise CorpusSnapshotError("CorpusIndex records must be a list")
    for record in records:
        if not isinstance(record, SentenceRecord):
            raise CorpusSnapshotError(
                "CorpusIndex contains a non-SentenceRecord value"
            )
        if not all(
            isinstance(value, str)
            for value in (
                record.original_sentence,
                record.normalized_sentence,
                record.source_text,
            )
        ):
            raise CorpusSnapshotError("SentenceRecord contains non-text fields")
        if not isinstance(record.offset, int):
            raise CorpusSnapshotError("SentenceRecord offset must be an integer")
    return records


def _validated_postings(
    raw_postings: dict[str, array],
    record_count: int,
) -> tuple[dict[str, array], int]:
    if not isinstance(raw_postings, dict):
        raise CorpusSnapshotError("CorpusIndex postings must be a dictionary")

    total_postings = 0
    for trigram, raw_sentence_ids in raw_postings.items():
        if not isinstance(trigram, str):
            raise CorpusSnapshotError("CorpusIndex contains a non-text trigram")
        if not isinstance(raw_sentence_ids, array):
            raise CorpusSnapshotError(
                f"Posting list for {trigram!r} is not an array"
            )
        if raw_sentence_ids.typecode != POSTING_TYPE_CODE:
            raise CorpusSnapshotError(
                f"Posting list for {trigram!r} has an incompatible type code"
            )
        _validate_posting_ids(trigram, raw_sentence_ids, record_count)
        total_postings += len(raw_sentence_ids)
    return raw_postings, total_postings


def _validate_posting_ids(
    trigram: str,
    sentence_ids: array,
    record_count: int,
) -> None:
    previous_id = -1
    for sentence_id in sentence_ids:
        if sentence_id >= record_count:
            raise SnapshotCorruptionError(
                f"Posting {trigram!r} contains out-of-range sentence ID "
                f"{sentence_id}"
            )
        if sentence_id == previous_id:
            raise SnapshotCorruptionError(
                f"Posting {trigram!r} contains duplicate sentence ID {sentence_id}"
            )
        if sentence_id < previous_id:
            raise SnapshotCorruptionError(
                f"Posting {trigram!r} is not sorted"
            )
        previous_id = sentence_id


def _encode_posting_ids(sentence_ids: array) -> bytes:
    if sentence_ids.itemsize != _POSTING_WIDTH_BYTES:
        raise SnapshotCompatibilityError(
            "This runtime does not provide four-byte array('I') postings"
        )
    if sys.byteorder == _POSTING_BYTE_ORDER:
        return sentence_ids.tobytes()
    portable = array(POSTING_TYPE_CODE, sentence_ids)
    portable.byteswap()
    return portable.tobytes()


def _decode_posting_ids(payload: bytes) -> array:
    if len(payload) % _POSTING_WIDTH_BYTES:
        raise SnapshotCorruptionError("Posting payload length is not divisible by four")
    sentence_ids = array(POSTING_TYPE_CODE)
    if sentence_ids.itemsize != _POSTING_WIDTH_BYTES:
        raise SnapshotCompatibilityError(
            "This runtime does not provide four-byte array('I') postings"
        )
    sentence_ids.frombytes(payload)
    if sys.byteorder != _POSTING_BYTE_ORDER:
        sentence_ids.byteswap()
    return sentence_ids


def _metadata_nonnegative_int(metadata: dict[str, str], key: str) -> int:
    try:
        value = int(metadata[key])
    except ValueError as error:
        raise SnapshotCorruptionError(
            f"Snapshot metadata {key!r} is not an integer"
        ) from error
    if value < 0:
        raise SnapshotCorruptionError(
            f"Snapshot metadata {key!r} cannot be negative"
        )
    return value


def _metadata_build_seconds(metadata: dict[str, str]) -> float:
    try:
        value = float(metadata["build_seconds"])
    except ValueError as error:
        raise SnapshotCorruptionError("Snapshot build time is invalid") from error
    if not math.isfinite(value) or value < 0:
        raise SnapshotCorruptionError("Snapshot build time is invalid")
    return value


def _validated_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Archive SHA-256 must be a hexadecimal string")
    normalized = value.lower()
    if len(normalized) != 64:
        raise ValueError("Archive SHA-256 must contain 64 hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(
            "Archive SHA-256 must contain 64 hexadecimal characters"
        )
    return normalized


def _snapshot_sha256(value: str) -> str:
    try:
        return _validated_sha256(value)
    except ValueError as error:
        raise SnapshotCorruptionError(
            "Snapshot archive fingerprint is invalid"
        ) from error


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
