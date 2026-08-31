"""Persistent embedding cache (Spec 1).

Embedding the corpus costs time, money and API quota, so the result is
written to disk once and reloaded on every later run.

Layout
------

``path`` is a DIRECTORY holding three files:

    manifest.json     {"version": 2, "count": 3, "dim": 768, ...}
    metadata.jsonl    one JSON object per sentence, in order
    metadata.idx      one int64 per sentence: the END byte offset of its
                      metadata line, so line i can be reached with a seek
                      instead of by reading everything before it
    vectors.f32       count * dim raw float32 numbers, back to back

Exactly one format
------------------

This is the same directory that ``EmbeddingStore`` memory maps for the
full corpus.  Small samples are read eagerly by ``load_embeddings`` into
``list[EmbeddedSentence]``; a corpus too large for RAM is streamed by the
store.  Two ways of reading, one thing on disk.

Why the vectors are binary float32
----------------------------------

A 768 number vector written as JSON text costs about 20 bytes per number;
written as a raw float32 it costs exactly 4.  Over the full corpus of
roughly 2.58 million sentences that is the difference between about 40 GB
and about 8 GB.  The sentence text itself stays in readable JSONL, so the
cache can still be inspected by eye.

float32 is the deliberate choice, and it is LOSSY.  A Python float is a
float64, so a value handed to ``save_embeddings`` comes back very slightly
different: 0.12 returns as 0.11999999731779099.  That is a relative error
of about 1e-7, which is far below the level at which two sentences differ
in meaning, and cosine similarity is unaffected in any way that could
reorder results.  Embeddings are approximate quantities, not identifiers,
so half the bytes is worth more than exact equality here.

Callers must therefore compare embeddings approximately, never with ``==``.

This mirrors what Part A already does in ``src/corpus/index.py``, where
posting lists are ``array('I')`` instead of ``set[int]`` for the same
reason.

Corruption
----------

The manifest records how many vectors of what size were written, so a
truncated or edited cache is caught by comparing the manifest against the
files themselves rather than by trusting whatever is on disk.
"""

from __future__ import annotations

import json
import sys
from array import array
from dataclasses import asdict
from pathlib import Path

from src.semantic.contracts import EmbeddedSentence


# Bumped if the on disk layout ever changes shape.
CACHE_VERSION = 2

# Type code of the metadata index: signed 64 bit byte offsets.
OFFSET_TYPE_CODE = "q"

# Type code of the vector file: C float, 4 bytes per number.  Narrower
# than a Python float, so values are rounded on the way to disk.
VECTOR_TYPE_CODE = "f"

MANIFEST_FILE_NAME = "manifest.json"
METADATA_FILE_NAME = "metadata.jsonl"
OFFSETS_FILE_NAME = "metadata.idx"
VECTORS_FILE_NAME = "vectors.f32"

METADATA_FIELDS = ("sentence", "source_text", "offset")


class CacheError(RuntimeError):
    """Raised when an embedding cache is missing, invalid or corrupted."""


def cache_files(path: str) -> tuple[Path, Path, Path, Path]:
    """Return the manifest, metadata, offsets and vector paths."""
    directory = Path(path)

    return (
        directory / MANIFEST_FILE_NAME,
        directory / METADATA_FILE_NAME,
        directory / OFFSETS_FILE_NAME,
        directory / VECTORS_FILE_NAME,
    )


def metadata_line(item: EmbeddedSentence) -> bytes:
    """Return the metadata JSONL line for one embedded sentence."""
    record = asdict(item)
    record.pop("embedding")

    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def build_manifest(count: int, dimension: int, complete: bool) -> dict:
    """Return the manifest describing a cache directory."""
    return {
        "version": CACHE_VERSION,
        "count": count,
        "dim": dimension,
        "dtype": "float32",
        "byteorder": sys.byteorder,
        "complete": complete,
    }


def write_manifest(path: str, manifest: dict) -> None:
    """Write the manifest, replacing any previous one."""
    manifest_path = cache_files(path)[0]
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str) -> dict:
    """Read and validate the manifest of a cache directory."""
    manifest_path = cache_files(path)[0]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheError(f"Unreadable cache manifest: {error}") from error

    if not isinstance(manifest, dict):
        raise CacheError("Cache manifest is not a JSON object")

    if manifest.get("version") != CACHE_VERSION:
        raise CacheError(
            f"Unsupported cache version: {manifest.get('version')!r}"
        )

    if not isinstance(manifest.get("count"), int):
        raise CacheError("Cache manifest is missing a valid count")

    if not isinstance(manifest.get("dim"), int):
        raise CacheError("Cache manifest is missing a valid dim")

    stored_byteorder = manifest.get("byteorder")

    if stored_byteorder != sys.byteorder:
        raise CacheError(
            f"Cache was written on a {stored_byteorder}-endian machine, "
            f"this one is {sys.byteorder}-endian"
        )

    return manifest

def cache_exists(path: str) -> bool:
    """Return whether every cache file is present."""
    return all(file_path.is_file() for file_path in cache_files(path))


def save_embeddings(items: list[EmbeddedSentence], path: str) -> None:
    """Write embedded sentences to the cache directory ``path``.

    This is the eager small sample writer.  It holds everything it is
    given in RAM, so it is meant for samples, not for the whole corpus;
    ``EmbeddingStoreWriter`` streams instead.
    """
    dimension = len(items[0].embedding) if items else 0

    for item in items:
        if len(item.embedding) != dimension:
            raise CacheError(
                "All embeddings must have the same length, found "
                f"{len(item.embedding)} and {dimension}"
            )

    _, metadata_path, offsets_path, vectors_path = cache_files(path)
    Path(path).mkdir(parents=True, exist_ok=True)

    offsets = array(OFFSET_TYPE_CODE)
    position = 0

    with metadata_path.open("wb") as metadata_file:
        for item in items:
            line = metadata_line(item)
            metadata_file.write(line)
            position += len(line)
            offsets.append(position)

    with offsets_path.open("wb") as offsets_file:
        offsets.tofile(offsets_file)

    with vectors_path.open("wb") as vectors_file:
        for item in items:
            array(VECTOR_TYPE_CODE, item.embedding).tofile(vectors_file)

    write_manifest(path, build_manifest(len(items), dimension, complete=True))


def read_metadata_lines(path: str) -> list[dict]:
    """Return one validated metadata dictionary per cached sentence."""
    metadata_path = cache_files(path)[1]

    try:
        text = metadata_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CacheError(f"Unreadable cache metadata: {error}") from error

    return [
        parse_metadata_line(line, number)
        for number, line in enumerate(text.splitlines(), start=1)
    ]


def parse_metadata_line(line: str, number: int) -> dict:
    """Validate one metadata line and return it as a dictionary."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise CacheError(
            f"Corrupted cache metadata on line {number}: {error}"
        ) from error

    if not isinstance(record, dict):
        raise CacheError(f"Cache metadata line {number} is not a JSON object")

    missing = [field for field in METADATA_FIELDS if field not in record]

    if missing:
        raise CacheError(
            f"Cache metadata line {number} is missing {', '.join(missing)}"
        )

    if not isinstance(record["offset"], int):
        raise CacheError(
            f"Cache metadata line {number} has a non integer offset"
        )

    return record


def read_offsets(path: str, count: int) -> array:
    """Return the end byte offset of every metadata line."""
    offsets_path = cache_files(path)[2]
    offsets = array(OFFSET_TYPE_CODE)

    try:
        with offsets_path.open("rb") as offsets_file:
            offsets.fromfile(offsets_file, count)
    except OSError as error:
        raise CacheError(f"Unreadable cache offsets: {error}") from error
    except EOFError as error:
        raise CacheError(
            f"Cache offset file is truncated: expected {count} offsets"
        ) from error

    if offsets_path.stat().st_size != count * offsets.itemsize:
        raise CacheError("Cache offset file size does not match the manifest")

    return offsets


def _load_vectors(path: str, count: int, dim: int) -> array:
    """Return every cached number as one flat float32 array."""
    vectors_path = cache_files(path)[3]
    expected_numbers = count * dim
    vectors = array(VECTOR_TYPE_CODE)

    try:
        with vectors_path.open("rb") as vectors_file:
            vectors.fromfile(vectors_file, expected_numbers)
    except OSError as error:
        raise CacheError(f"Unreadable cache vectors: {error}") from error
    except EOFError as error:
        raise CacheError(
            f"Cache vector file is truncated: expected {expected_numbers} "
            "numbers"
        ) from error

    if vectors_path.stat().st_size != expected_numbers * vectors.itemsize:
        raise CacheError("Cache vector file size does not match the manifest")

    return vectors


def load_embeddings(path: str) -> list[EmbeddedSentence]:
    """Read embedded sentences back from the cache directory ``path``.

    Everything is loaded into RAM, so this is for samples.  Use
    ``EmbeddingStore`` for a corpus that does not fit.
    """
    if not cache_exists(path):
        raise CacheError(f"No embedding cache found at {path}")

    manifest = read_manifest(path)
    count = manifest["count"]
    dimension = manifest["dim"]

    if not manifest.get("complete", True):
        raise CacheError(
            f"Cache at {path} is an unfinished build holding {count} "
            "sentences; finish or delete it"
        )

    metadata = read_metadata_lines(path)

    if len(metadata) != count:
        raise CacheError(
            f"Cache manifest promises {count} sentences but metadata holds "
            f"{len(metadata)}"
        )

    read_offsets(path, count)
    vectors = _load_vectors(path, count, dimension)

    return [
        EmbeddedSentence(
            sentence=record["sentence"],
            source_text=record["source_text"],
            offset=record["offset"],
            embedding=list(
                vectors[position * dimension:(position + 1) * dimension]
            ),
        )
        for position, record in enumerate(metadata)
    ]
