"""Full corpus embedding storage (Spec 1).

``load_embeddings`` reads a whole cache into RAM.  That is fine for a
sample and impossible for the real corpus: 2.58 million sentences of 768
numbers each would cost roughly 63 GB as Python lists, because every
number becomes a separate heap object.

This module reads the SAME cache directory without ever doing that.

    EmbeddingStore        read side.  Memory maps vectors.f32, so the
                          operating system pages vectors in on demand and
                          Python never holds more than one block.

    EmbeddingStoreWriter  write side.  Appends batch by batch and records
                          progress in the manifest, so an interrupted
                          build resumes instead of starting over.

Reading in three ways
---------------------

    store.vectors()       the whole thing as one (count, dim) float32
                          NumPy array, backed by the file.  Nothing is
                          read until it is touched.

    store.iter_blocks()   the same data in chunks, for scoring a corpus
                          larger than RAM a block at a time.

    iter(store)           one EmbeddedSentence at a time, matching the
                          shared contract exactly.  This is the seam that
                          lets semantic search accept either a plain list
                          or a corpus that does not fit in memory.

Crash safety
------------

Each batch is written vectors first, then metadata, then offsets, and the
manifest LAST.  The manifest is therefore never ahead of the data.  If a
build dies mid batch, reopening truncates the three data files back to
the count the manifest promises and carries on from there.
"""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Iterator

import numpy as np

from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import (
    CacheError,
    OFFSET_TYPE_CODE,
    VECTOR_TYPE_CODE,
    build_manifest,
    cache_files,
    metadata_line,
    parse_metadata_line,
    read_manifest,
    read_offsets,
    write_manifest,
)


# NumPy view of the vector file.  Must match VECTOR_TYPE_CODE ("f").
VECTOR_DTYPE = np.float32

# Bytes per stored number, and per stored byte offset.
VECTOR_ITEM_SIZE = 4
OFFSET_ITEM_SIZE = 8

# How many sentences to hand over at once by default.  4096 vectors of
# 768 float32 numbers is about 12 MB, which stays comfortably in cache.
DEFAULT_BLOCK_SIZE = 4096


class EmbeddingStore:
    """Read only, memory mapped access to an embedding cache directory."""

    def __init__(self, path: str):
        manifest = read_manifest(path)

        self.path = path
        self.count: int = manifest["count"]
        self.dim: int = manifest["dim"]
        self.is_complete: bool = bool(manifest.get("complete", True))

        self._vectors: np.ndarray | None = None
        self._offset_array: array | None = None

        self._verify_file_sizes()

    def _verify_file_sizes(self) -> None:
        """Check the files on disk against what the manifest promises."""
        _, _, offsets_path, vectors_path = cache_files(self.path)

        expected_vector_bytes = self.count * self.dim * VECTOR_ITEM_SIZE

        if vectors_path.stat().st_size != expected_vector_bytes:
            raise CacheError(
                "Cache vector file size does not match the manifest"
            )

        if offsets_path.stat().st_size != self.count * OFFSET_ITEM_SIZE:
            raise CacheError(
                "Cache offset file size does not match the manifest"
            )

    def __len__(self) -> int:
        return self.count

    def vectors(self) -> np.ndarray:
        """Return every vector as one memory mapped (count, dim) array.

        No vector is read from disk until it is actually used, so this is
        cheap even for a file of several gigabytes.
        """
        if self.count == 0:
            return np.empty((0, self.dim), dtype=VECTOR_DTYPE)

        if self._vectors is None:
            self._vectors = np.memmap(
                cache_files(self.path)[3],
                dtype=VECTOR_DTYPE,
                mode="r",
                shape=(self.count, self.dim),
            )

        return self._vectors

    def iter_blocks(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(start_index, block)`` pairs covering every vector.

        ``block`` is a view into the memory map, not a copy.
        """
        if block_size < 1:
            raise ValueError("block_size must be at least 1")

        vectors = self.vectors()

        for start in range(0, self.count, block_size):
            yield start, vectors[start:start + block_size]

    def _offsets(self) -> array:
        """Return the metadata line offsets, reading them only once."""
        if self._offset_array is None:
            self._offset_array = read_offsets(self.path, self.count)

        return self._offset_array

    def metadata(self, index: int) -> dict:
        """Return the stored sentence, source and offset of one record."""
        if not 0 <= index < self.count:
            raise IndexError(f"No sentence at index {index}")

        offsets = self._offsets()
        start = offsets[index - 1] if index > 0 else 0
        length = offsets[index] - start

        with cache_files(self.path)[1].open("rb") as metadata_file:
            metadata_file.seek(start)
            raw = metadata_file.read(length)

        return parse_metadata_line(raw.decode("utf-8").strip(), index + 1)

    def get(self, index: int) -> EmbeddedSentence:
        """Return one full record, used to turn a hit into a result."""
        record = self.metadata(index)

        return EmbeddedSentence(
            sentence=record["sentence"],
            source_text=record["source_text"],
            offset=record["offset"],
            embedding=self.vectors()[index].tolist(),
        )

    def __iter__(self) -> Iterator[EmbeddedSentence]:
        """Yield every record in order, one at a time.

        Memory stays flat: each vector is converted, handed over, and
        then left for the garbage collector.
        """
        vectors = self.vectors()
        metadata_path = cache_files(self.path)[1]

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for index, line in enumerate(metadata_file):
                if index >= self.count:
                    break

                record = parse_metadata_line(line.strip(), index + 1)

                yield EmbeddedSentence(
                    sentence=record["sentence"],
                    source_text=record["source_text"],
                    offset=record["offset"],
                    embedding=vectors[index].tolist(),
                )

    def close(self) -> None:
        """Release the memory map."""
        if self._vectors is not None:
            self._vectors._mmap.close()
            self._vectors = None

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


class EmbeddingStoreWriter:
    """Append embedded sentences to a cache directory, batch by batch."""

    def __init__(self, path: str, dim: int):
        self.path = path
        self.dim = dim

        Path(path).mkdir(parents=True, exist_ok=True)

        self.count = self._resume()
        self._metadata_bytes = self._metadata_end()

        _, metadata_path, offsets_path, vectors_path = cache_files(path)
        self._metadata_file = metadata_path.open("ab")
        self._offsets_file = offsets_path.open("ab")
        self._vectors_file = vectors_path.open("ab")

    def _resume(self) -> int:
        """Return how many records survive, discarding any partial batch."""
        manifest_path, *data_paths = cache_files(self.path)

        for file_path in data_paths:
            if not file_path.exists():
                file_path.write_bytes(b"")

        if not manifest_path.is_file():
            write_manifest(
                self.path,
                build_manifest(0, self.dim, complete=False),
            )
            return 0

        manifest = read_manifest(self.path)
        count = manifest["count"]

        if count > 0 and manifest["dim"] != self.dim:
            raise CacheError(
                f"Cache at {self.path} holds {manifest['dim']} dimensional "
                f"vectors, cannot append {self.dim} dimensional ones"
            )

        self._truncate_to(count)

        return count

    def _truncate_to(self, count: int) -> None:
        """Cut the data files back to exactly ``count`` complete records."""
        _, metadata_path, offsets_path, vectors_path = cache_files(self.path)

        with offsets_path.open("r+b") as offsets_file:
            offsets_file.truncate(count * OFFSET_ITEM_SIZE)

        offsets = read_offsets(self.path, count)
        metadata_bytes = offsets[count - 1] if count > 0 else 0

        with metadata_path.open("r+b") as metadata_file:
            metadata_file.truncate(metadata_bytes)

        with vectors_path.open("r+b") as vectors_file:
            vectors_file.truncate(count * self.dim * VECTOR_ITEM_SIZE)

    def _metadata_end(self) -> int:
        """Return the byte length of the metadata written so far."""
        if self.count == 0:
            return 0

        return read_offsets(self.path, self.count)[self.count - 1]

    def append(self, items: list[EmbeddedSentence]) -> None:
        """Write one batch, then record it in the manifest."""
        for item in items:
            if len(item.embedding) != self.dim:
                raise CacheError(
                    f"Expected {self.dim} dimensional embeddings, got "
                    f"{len(item.embedding)}"
                )

        if not items:
            return

        for item in items:
            array(VECTOR_TYPE_CODE, item.embedding).tofile(self._vectors_file)

        offsets = array(OFFSET_TYPE_CODE)

        for item in items:
            line = metadata_line(item)
            self._metadata_file.write(line)
            self._metadata_bytes += len(line)
            offsets.append(self._metadata_bytes)

        offsets.tofile(self._offsets_file)

        for handle in self._handles():
            handle.flush()

        self.count += len(items)

        # Written last, so the manifest is never ahead of the data.
        write_manifest(
            self.path,
            build_manifest(self.count, self.dim, complete=False),
        )

    def finish(self) -> None:
        """Mark the build complete and close the files."""
        write_manifest(
            self.path,
            build_manifest(self.count, self.dim, complete=True),
        )
        self.close()

    def _handles(self) -> tuple:
        """Return the three open data file handles."""
        return (self._vectors_file, self._metadata_file, self._offsets_file)

    def close(self) -> None:
        """Close the open file handles."""
        for handle in self._handles():
            if not handle.closed:
                handle.close()

    def __enter__(self) -> "EmbeddingStoreWriter":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
