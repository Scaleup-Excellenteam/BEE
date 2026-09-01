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
from src.semantic.embedding_cache import CacheError, cache_files
from src.semantic.embedding_store import EmbeddingStore, EmbeddingStoreWriter
from src.semantic.local_provider import (
    DEFAULT_BATCH_SIZE,
    embed_text,
    embed_texts,
    warm_up,
)
from src.semantic.search import semantic_search
from src.logsearch.log_parser import (
    FAULT_LEVELS,
    LogRecord,
    parse_fault_records,
    records_after,
)


# The satellite's fault history.  This is deliberately NOT
# logs/autocomplete.log: that file is the application's own runtime
# diary, and mixing "user submitted a search query" into a satellite
# fault history would make every search meaningless.
DEFAULT_LOG_PATH = "logs/satellite_faults.log"

# Faults collected from previous satellites, previous missions and test
# environments.  Imported into the history once, on the first run.
BOOTSTRAP_DATASET_PATH = "data/historical_satellite_faults.log"

# Kept apart from the corpus caches on purpose.  Those hold 768 wide
# Gemini vectors; these are 384 wide local ones, and mixing the two would
# be rejected by the store anyway.
DEFAULT_CACHE_PATH = "satellite_fault_cache"

# Matches the format in src/logging_config.py, so appended lines parse
# back exactly like the ones logging wrote.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cosine similarity a historical fault must reach to count as a match.
#
# Ranking alone is not enough: the nearest of a thousand unrelated faults
# is still unrelated, and showing it as a "similar fault" is worse than
# showing nothing.  Observed on real data, "hi" scored about 0.08 and
# "error" about 0.40 against faults that had nothing to do with them, so
# the line is drawn at 0.35.  Below it, the honest answer is "no match".
MIN_LOG_SIMILARITY = 0.35


class LogSearchService:
    """Searchable, incrementally maintained index of historical faults."""

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        cache_path: str = DEFAULT_CACHE_PATH,
        dataset_path: str = BOOTSTRAP_DATASET_PATH,
        embedder=embed_text,
        batch_embedder=embed_texts,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_similarity: float = MIN_LOG_SIMILARITY,
        warm_up_fn=warm_up,
    ):
        self.log_path = Path(log_path)
        self.cache_path = cache_path
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.embedder = embedder
        self.batch_embedder = batch_embedder
        self.batch_size = batch_size
        self.min_similarity = min_similarity
        self.warm_up_fn = warm_up_fn

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
            self._verify_cache_matches_log()

    def _verify_cache_matches_log(self) -> None:
        """Refuse a cache that was built from a different history file.

        Silently reusing the wrong cache would answer satellite queries
        out of some other log, which is worse than failing loudly.
        """
        if self._store is None or len(self._store) == 0:
            return

        stored_source = self._store.metadata(0)["source_text"]

        if stored_source != self.log_path.name:
            raise CacheError(
                f"The cache at {self.cache_path} was built from "
                f"{stored_source}, but this service reads "
                f"{self.log_path.name}; delete the cache or point at the "
                "right one"
            )

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

        if last_offset > 0 and (
            not history or history[-1].offset < last_offset
        ):
            raise CacheError(
                f"The cache at {self.cache_path} already holds a record at "
                f"line {last_offset}, which {self.log_path} no longer has; "
                "the history was truncated or replaced"
            )

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

    def warm_up(self) -> None:
        """Load the embedding model now rather than on the first query.

        This embeds one throwaway token, never any log text, so calling
        it after ``refresh`` does not embed the history a second time.
        """
        self.warm_up_fn()

    def bootstrap(self) -> int:
        """Seed the satellite history from the prepared dataset, once.

        The dataset holds faults from previous satellites, previous
        missions and test environments.  It is copied into the history
        file only when that history does not exist yet, so a restart
        never imports it a second time and never duplicates a record.
        Runtime faults are appended to the same file afterwards, which
        is what keeps prepared and learned faults in ONE history.

        Returns how many fault records were imported.
        """
        if self.dataset_path is None or not self.dataset_path.is_file():
            return 0

        if self.log_path.is_file() and self.log_path.stat().st_size > 0:
            return 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            self.dataset_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        return len(parse_fault_records(self.log_path))

    def refresh(self) -> int:
        """Import any prepared history, then index whatever is new.

        Returns how many new records were embedded.  On a first run this
        imports the dataset and builds the whole cache; afterwards it
        normally does nothing.
        """
        self.bootstrap()

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
        """Return historical faults genuinely similar to ``error_text``.

        At most ``k`` results come back, and possibly none.  Ranking is
        left entirely to ``semantic_search``; this only drops results
        that are too weak to be worth showing, so a query with no real
        match returns an empty list instead of five bad guesses.
        """
        if self._store is None or len(self._store) == 0:
            return []

        ranked = semantic_search(
            error_text,
            self._store,
            self.embedder,
            k=k,
        )

        # ``ranked`` is already ordered best first, so filtering keeps
        # the strongest matches and simply stops including the weak tail.
        return [
            result
            for result in ranked
            if result.similarity >= self.min_similarity
        ][:k]

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

        # An empty entry would append a blank line to the history and
        # embed nothing meaningful, so it is refused outright.
        if not error_text.strip():
            raise ValueError("A fault must have a message")

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
