"""Incremental semantic search over historical fault logs.

What this is for
----------------

When the satellite hits an error, the useful question is "has this gone
wrong before, and what did it look like?".  This module answers that by
comparing the new error against every ERROR, WARNING and CRITICAL entry
already in the log.

Runtime order for a new error
-----------------------------

    new error
        -> embed it locally
        -> search ONLY the historical cache
        -> Top 5 similar past errors
        -> close the EmbeddingStore
        -> append the new error to the log, then to the cache
        -> reopen the store

The search happens BEFORE the append, so a new error can never be
returned as a match for itself.  That ordering is the whole point of
``record_error`` and is checked by a test.

Closing the store around the append is not decoration.  ``EmbeddingStore``
holds ``vectors.f32`` open through ``numpy.memmap``; on Windows, writing
to a file that is still mapped can fail with a sharing violation.  So the
lifecycle is always search -> close -> append -> reopen.

Startup
-------

The first run parses the log's fault history, embeds it and creates the
cache.  Every later run reuses that cache and embeds ONLY the entries
appended since, found by comparing line offsets.  Because the log file is
append only, a line number never changes, so "everything past the last
indexed offset" is exactly the set of new records.

Nothing here reads ``input()`` or prints, so a satellite process can call
it directly.
"""

from __future__ import annotations

from datetime import datetime
from itertools import batched
from pathlib import Path

from src.semantic.contracts import EmbeddedSentence, SemanticResult
from src.semantic.embedding_cache import cache_files
from src.semantic.embedding_store import EmbeddingStore, EmbeddingStoreWriter
from src.semantic.local_provider import (
    DEFAULT_BATCH_SIZE,
    embed_text,
    embed_texts,
)
from src.semantic.search import semantic_search
from src.logsearch.log_parser import (
    FAULT_LEVELS,
    LogRecord,
    parse_fault_records,
    records_after,
)


# The log written by src/logging_config.py.
DEFAULT_LOG_PATH = "logs/autocomplete.log"

# Kept apart from the corpus caches on purpose.  Those hold 768 wide
# Gemini vectors; these are 384 wide local ones, and mixing the two would
# be rejected by the store anyway.
DEFAULT_CACHE_PATH = "log_embedding_cache"

# Matches the format in src/logging_config.py, so appended lines parse
# back exactly like the ones logging wrote.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogSearchService:
    """Searchable, incrementally maintained index of historical faults."""

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        cache_path: str = DEFAULT_CACHE_PATH,
        embedder=embed_text,
        batch_embedder=embed_texts,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.log_path = Path(log_path)
        self.cache_path = cache_path
        self.embedder = embedder
        self.batch_embedder = batch_embedder
        self.batch_size = batch_size

        self._store: EmbeddingStore | None = None
        self._open_store()

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _cache_built(self) -> bool:
        """Return whether a cache already exists on disk."""
        return cache_files(self.cache_path)[0].is_file()

    def _open_store(self) -> None:
        """Open the cache for reading, if it has been built."""
        if self._store is None and self._cache_built():
            self._store = EmbeddingStore(self.cache_path)

    def _close_store(self) -> None:
        """Release the memory map so the files can be appended to."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def close(self) -> None:
        """Release every open resource."""
        self._close_store()

    def __enter__(self) -> "LogSearchService":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._store) if self._store is not None else 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def last_indexed_offset(self) -> int:
        """Return the log line number of the newest indexed record."""
        if self._store is None or len(self._store) == 0:
            return 0

        return self._store.metadata(len(self._store) - 1)["offset"]

    def _pending_records(self, last_offset: int) -> list[LogRecord]:
        """Return the fault records that start after ``last_offset``.

        The offset is passed in rather than read here, because callers
        must read it while the store is still OPEN: a closed store
        reports no offset at all, which would look like an empty cache
        and re-embed the whole history.
        """
        history = parse_fault_records(self.log_path)

        return records_after(history, last_offset)

    def _index(self, records: list[LogRecord]) -> None:
        """Embed records and append them to the cache.

        The caller is responsible for having closed the store first.
        """
        writer = None

        try:
            for batch in batched(records, self.batch_size):
                vectors = self.batch_embedder(
                    [record.message for record in batch]
                )
                items = [
                    EmbeddedSentence(
                        sentence=record.message,
                        source_text=record.source_text,
                        offset=record.offset,
                        embedding=vector,
                    )
                    for record, vector in zip(batch, vectors)
                ]

                if writer is None:
                    writer = EmbeddingStoreWriter(
                        self.cache_path,
                        dim=len(items[0].embedding),
                    )

                writer.append(items)

            if writer is not None:
                writer.finish()
        finally:
            if writer is not None:
                writer.close()

    def refresh(self) -> int:
        """Index every fault entry appended since the last run.

        Returns how many new records were embedded.  On a first run this
        builds the whole cache; afterwards it normally does nothing.
        """
        pending = self._pending_records(self.last_indexed_offset())

        if not pending:
            return 0

        self._close_store()
        self._index(pending)
        self._open_store()

        return len(pending)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search_similar_logs(
        self,
        error_text: str,
        k: int = 5,
    ) -> list[SemanticResult]:
        """Return the ``k`` historical faults most similar to the text.

        This never writes anything, so it is safe to call as often as
        wanted.
        """
        if self._store is None or len(self._store) == 0:
            return []

        return semantic_search(
            error_text,
            self._store,
            self.embedder,
            k=k,
        )

    # ------------------------------------------------------------------
    # The full new-error flow
    # ------------------------------------------------------------------

    def _append_to_log(self, message: str, level: str) -> None:
        """Append one entry to the log file in the standard format."""
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} | {level} | {message}\n")

    def record_error(
        self,
        error_text: str,
        level: str = "ERROR",
        k: int = 5,
    ) -> list[SemanticResult]:
        """Search history for a new error, then record that error.

        Returns the Top ``k`` similar PREVIOUS faults.  The new error is
        appended only afterwards, so it is never matched against itself.
        """
        if level not in FAULT_LEVELS:
            raise ValueError(
                f"{level} is not a fault level; expected one of "
                f"{', '.join(sorted(FAULT_LEVELS))}"
            )

        results = self.search_similar_logs(error_text, k=k)

        # Read the watermark while the store is still open, then follow
        # the safe lifecycle: close, append, reopen.
        last_offset = self.last_indexed_offset()

        self._close_store()
        self._append_to_log(error_text, level)
        self._index(self._pending_records(last_offset))
        self._open_store()

        return results


_default_service: LogSearchService | None = None


def get_default_service() -> LogSearchService:
    """Return a shared service, building or updating its cache once."""
    global _default_service

    if _default_service is None:
        _default_service = LogSearchService()
        _default_service.refresh()

    return _default_service


def search_similar_logs(error_text: str, k: int = 5) -> list[SemanticResult]:
    """Return the ``k`` historical faults most similar to ``error_text``."""
    return get_default_service().search_similar_logs(error_text, k=k)


def record_error(
    error_text: str,
    level: str = "ERROR",
    k: int = 5,
) -> list[SemanticResult]:
    """Search history for a new error, then append it to log and cache."""
    return get_default_service().record_error(error_text, level=level, k=k)
