"""Build or reuse the semantic embedding index (Spec 1).

There are two entry points, for two very different sizes of job.

``load_or_build_embeddings`` -- small samples
--------------------------------------------

    records
        -> valid cache on disk?
             yes -> load it and stop, Gemini is never called
             no  -> embed each ORIGINAL sentence, one request per sentence
                    -> save the cache
        -> list[EmbeddedSentence]

Everything ends up in RAM, so this is for tests and small samples.

``open_or_build_store`` -- the full corpus
------------------------------------------

    records
        -> finished store on disk?
             yes -> open it and stop, Gemini is never called
             no  -> resume at whatever the manifest already has
                    -> embed the next BATCH of sentences in one request
                    -> append the batch to disk
                    -> repeat, so RAM never holds more than one batch
        -> EmbeddingStore

Batching is what makes the real corpus affordable: at 100 sentences per
request, 2.58 million sentences cost about 26,000 requests instead of
2.58 million.  Resuming is what makes it survivable, because a job that
size will be interrupted at least once.

In both cases the embedding function is a parameter rather than a hard
wired import, so tests can pass a fake and exercise the whole pipeline
with no network.
"""

from __future__ import annotations

from itertools import batched, islice
from typing import Callable, Iterable

from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import (
    cache_exists,
    cache_files,
    load_embeddings,
    read_manifest,
    save_embeddings,
)
from src.semantic.embedding_provider import (
    MAX_BATCH_SIZE,
    embed_text,
    embed_texts,
)
from src.semantic.embedding_store import EmbeddingStore, EmbeddingStoreWriter


# Where the cache lives unless the caller says otherwise.
DEFAULT_CACHE_PATH = "embedding_cache"

# What a replacement embedding function has to look like.
Embedder = Callable[[str], list[float]]
BatchEmbedder = Callable[[list[str]], list[list[float]]]

# Called with the running total after each batch, for long builds.
ProgressCallback = Callable[[int], None]


def build_embeddings(
    records: Iterable,
    embedder: Embedder = embed_text,
) -> list[EmbeddedSentence]:
    """Embed every record, keeping its real source text and offset."""
    return [
        EmbeddedSentence(
            sentence=record.original_sentence,
            source_text=record.source_text,
            offset=record.offset,
            embedding=embedder(record.original_sentence),
        )
        for record in records
    ]


def load_or_build_embeddings(
    records: Iterable,
    cache_path: str = DEFAULT_CACHE_PATH,
    embedder: Embedder = embed_text,
) -> list[EmbeddedSentence]:
    """Return embedded sentences, reusing the cache whenever it exists.

    A cache that is present but unreadable raises instead of quietly
    rebuilding: rebuilding means paying for the whole corpus again, which
    is not something this function should decide on its own.  Delete the
    cache directory to force a rebuild.
    """
    if cache_exists(cache_path):
        return load_embeddings(cache_path)

    items = build_embeddings(records, embedder)
    save_embeddings(items, cache_path)

    return items


def _finished_build(cache_path: str) -> dict | None:
    """Return the manifest of an existing build, or ``None`` if there is none."""
    if not cache_files(cache_path)[0].is_file():
        return None

    return read_manifest(cache_path)


def _batch_to_items(records: tuple, vectors: list[list[float]]) -> list:
    """Pair a batch of records with the vectors that came back for them."""
    return [
        EmbeddedSentence(
            sentence=record.original_sentence,
            source_text=record.source_text,
            offset=record.offset,
            embedding=vector,
        )
        for record, vector in zip(records, vectors)
    ]


def open_or_build_store(
    records: Iterable,
    cache_path: str = DEFAULT_CACHE_PATH,
    batch_embedder: BatchEmbedder = embed_texts,
    batch_size: int = MAX_BATCH_SIZE,
    progress: ProgressCallback | None = None,
) -> EmbeddingStore:
    """Return a memory mapped store, building or resuming it if needed.

    ``records`` must be in the same order on every run, because a resumed
    build skips the ones already on disk by position.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    manifest = _finished_build(cache_path)

    if manifest is not None and manifest.get("complete", True):
        return EmbeddingStore(cache_path)

    already_built = manifest["count"] if manifest else 0
    writer = None

    for batch in batched(islice(records, already_built, None), batch_size):
        vectors = batch_embedder(
            [record.original_sentence for record in batch]
        )
        items = _batch_to_items(batch, vectors)

        if writer is None:
            writer = EmbeddingStoreWriter(
                cache_path,
                dim=len(items[0].embedding),
            )

        writer.append(items)

        if progress is not None:
            progress(writer.count)

    if writer is None:
        # Nothing left to embed: finish whatever is already on disk.
        writer = EmbeddingStoreWriter(
            cache_path,
            dim=manifest["dim"] if manifest else 0,
        )

    writer.finish()

    return EmbeddingStore(cache_path)
