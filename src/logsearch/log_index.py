"""Semantic fault memory for a satellite.

What this is for
----------------

When the satellite hits a fault, the useful question is "has this gone
wrong before, and what did it look like?".  This module answers that by
comparing the new fault against every incident already recorded.

Two stores, one canonical
-------------------------

    logs/satellite_incidents.jsonl   the incidents.  SOURCE OF TRUTH.
    satellite_fault_cache/           one embedding per incident.  An
                                     optimization, rebuildable from the
                                     incidents at any time.

The cache is kept in step by position: incident ``n`` has vector ``n``.
If the cache is short, the missing tail is embedded; if it is longer than
the history, it is stale and the service refuses to use it.

Runtime order for a new fault
-----------------------------

    new fault
        -> embed it locally
        -> search ONLY the existing incidents
        -> Top 5 similar past incidents, filtered at MIN_LOG_SIMILARITY
        -> decide: is this a REPEAT of one of them?
             repeat -> count += 1, last_seen = now, NO new embedding
             new    -> append incident, embed it once, append the vector
        -> return the Top 5

The search happens BEFORE anything is written, so a new fault can never
be returned as a match for itself.

Two thresholds, two questions
-----------------------------

    MIN_LOG_SIMILARITY          0.35   "is this worth showing you?"
    DEDUP_SIMILARITY_THRESHOLD  0.90   "is this literally the same fault?"

They are deliberately far apart.  Being similar enough to display is a
much weaker claim than being the same incident, and merging two distinct
faults destroys information permanently.

Nothing here reads ``input()`` or prints, so a satellite process can call
it directly.
"""

from __future__ import annotations

from datetime import datetime
from itertools import batched
from pathlib import Path

from src.semantic.embedding_cache import CacheError, cache_files
from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_store import EmbeddingStore, EmbeddingStoreWriter
from src.semantic.local_provider import (
    DEFAULT_BATCH_SIZE,
    embed_text,
    embed_texts,
    warm_up,
)
from src.semantic.search import semantic_search
from src.logsearch.incident_history import (
    DEFAULT_INCIDENTS_PATH,
    IncidentHistory,
)
from src.logsearch.incidents import (
    DEDUP_SIMILARITY_THRESHOLD,
    DEFAULT_SEVERITY,
    DEFAULT_SUBSYSTEM,
    FaultIncident,
    IncidentMatch,
    SEVERITIES,
    find_duplicate,
    normalize_subsystem,
)
from src.logsearch.log_parser import parse_fault_records


# Faults collected from previous satellites, previous missions and test
# environments.  Migrated into incidents once, on the first run.
BOOTSTRAP_DATASET_PATH = "data/historical_satellite_faults.log"

# The plain fault log written by earlier versions.  Still migrated on a
# first run so an existing deployment keeps its history, but no longer
# written to: incidents are the history now.
LEGACY_LOG_PATH = "logs/satellite_faults.log"

# Kept apart from the corpus caches on purpose.  Those hold 768 wide
# Gemini vectors; these are 384 wide local ones.
DEFAULT_CACHE_PATH = "satellite_fault_cache"

# Written into the cache metadata so a cache built for something else is
# recognised rather than silently trusted.
CACHE_SOURCE_NAME = "satellite_incidents"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cosine similarity a past incident must reach to be worth showing.
#
# Ranking alone is not enough: the nearest of a thousand unrelated faults
# is still unrelated, and showing it as a "similar fault" is worse than
# showing nothing.  Observed on real data, "hi" scored about 0.08 and
# "error" about 0.40 against faults that had nothing to do with them, so
# the line is drawn at 0.35.  Below it, the honest answer is "no match".
MIN_LOG_SIMILARITY = 0.35

# How many ranked candidates the deduplication check may consider.  It
# only ever looks at candidates at or above 0.90, and a deeper list than
# the displayed Top 5 means a near-identical incident is still found even
# when several unrelated subsystems score higher.
DEDUP_CANDIDATE_COUNT = 25


class LogSearchService:
    """Searchable, incrementally maintained memory of satellite faults."""

    def __init__(
        self,
        incidents_path: str = DEFAULT_INCIDENTS_PATH,
        cache_path: str = DEFAULT_CACHE_PATH,
        dataset_path: str | None = BOOTSTRAP_DATASET_PATH,
        legacy_log_path: str | None = LEGACY_LOG_PATH,
        embedder=embed_text,
        batch_embedder=embed_texts,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_similarity: float = MIN_LOG_SIMILARITY,
        dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
        warm_up_fn=warm_up,
        now_fn=None,
    ):
        self.history = IncidentHistory(incidents_path)
        self.cache_path = cache_path
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.legacy_log_path = (
            Path(legacy_log_path) if legacy_log_path else None
        )
        self.embedder = embedder
        self.batch_embedder = batch_embedder
        self.batch_size = batch_size
        self.min_similarity = min_similarity
        self.dedup_threshold = dedup_threshold
        self.warm_up_fn = warm_up_fn
        self.now_fn = now_fn or (
            lambda: datetime.now().strftime(TIMESTAMP_FORMAT)
        )

        self.incidents: list[FaultIncident] = self.history.load()
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
            self._verify_cache_matches_history()

    def _verify_cache_matches_history(self) -> None:
        """Refuse a cache that does not belong to this history.

        Silently reusing the wrong cache would answer satellite queries
        out of some other corpus, which is worse than failing loudly.
        """
        if self._store is None or len(self._store) == 0:
            return

        stored_source = self._store.metadata(0)["source_text"]

        if stored_source != CACHE_SOURCE_NAME:
            raise CacheError(
                f"The cache at {self.cache_path} was built from "
                f"{stored_source}, not {CACHE_SOURCE_NAME}; delete it or "
                "point at the right one"
            )

        if len(self._store) > len(self.incidents):
            raise CacheError(
                f"The cache at {self.cache_path} holds {len(self._store)} "
                f"vectors but the history has only {len(self.incidents)} "
                "incidents; the history was truncated or replaced"
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
        return len(self.incidents)

    # ------------------------------------------------------------------
    # Bootstrap and indexing
    # ------------------------------------------------------------------

    def _migration_source(self) -> Path | None:
        """Return the file a first run should build incidents from.

        An existing plain fault log wins over the prepared dataset,
        because an earlier version already copied the dataset into it and
        may have appended real faults after that.
        """
        if (
            self.legacy_log_path is not None
            and self.legacy_log_path.is_file()
            and self.legacy_log_path.stat().st_size > 0
        ):
            return self.legacy_log_path

        if self.dataset_path is not None and self.dataset_path.is_file():
            return self.dataset_path

        return None

    def bootstrap(self) -> int:
        """Build the incident history once, from the prepared dataset.

        Log-style lines carry no structure, so they migrate with honest
        defaults: the parsed level becomes the severity, the parsed
        timestamp becomes both first and last seen, the subsystem is
        UNKNOWN rather than guessed, and each starts at a count of one.

        Returns how many incidents were created.  Later runs create none,
        because the history file already exists.
        """
        if self.history.exists():
            return 0

        source = self._migration_source()

        if source is None:
            return 0

        records = parse_fault_records(source)

        self.incidents = [
            FaultIncident(
                incident_id=position,
                message=record.message,
                subsystem=DEFAULT_SUBSYSTEM,
                severity=record.level,
                error_code=None,
                first_seen=record.timestamp,
                last_seen=record.timestamp,
                count=1,
                source_text=record.source_text,
                source_offset=record.offset,
            )
            for position, record in enumerate(records, start=1)
        ]

        self.history.save(self.incidents)

        return len(self.incidents)

    def _indexed_count(self) -> int:
        """Return how many incidents already have an embedding."""
        return len(self._store) if self._store is not None else 0

    def _index(self, incidents: list[FaultIncident]) -> None:
        """Embed incidents and append them to the cache.

        The caller is responsible for having closed the store first.
        """
        if not incidents:
            return

        writer = None

        try:
            for batch in batched(incidents, self.batch_size):
                vectors = self.batch_embedder(
                    [incident.message for incident in batch]
                )
                items = [
                    EmbeddedSentence(
                        sentence=incident.message,
                        source_text=CACHE_SOURCE_NAME,
                        offset=incident.incident_id,
                        embedding=vector,
                    )
                    for incident, vector in zip(batch, vectors)
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
        """Import any prepared history, then embed whatever is missing.

        Returns how many incidents were newly embedded.  On a first run
        that is the whole migrated history; afterwards it is normally
        zero, because every incident already has its vector.
        """
        self.bootstrap()

        pending = self.incidents[self._indexed_count():]

        if not pending:
            return 0

        self._close_store()
        self._index(pending)
        self._open_store()

        return len(pending)

    def warm_up(self) -> None:
        """Load the embedding model now rather than on the first query.

        This embeds one throwaway token, never any incident text, so
        calling it after ``refresh`` embeds nothing a second time.
        """
        self.warm_up_fn()

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def _incident_by_id(self, incident_id: int) -> FaultIncident | None:
        """Return the incident with this id, if the history still has it."""
        if 1 <= incident_id <= len(self.incidents):
            return self.incidents[incident_id - 1]

        return None

    def _rank(self, text: str, k: int) -> list[IncidentMatch]:
        """Return the ``k`` most similar incidents, unfiltered.

        Ordering comes entirely from ``semantic_search``.  Nothing here
        reorders by count, severity or recency.
        """
        if self._store is None or len(self._store) == 0:
            return []

        matches = []

        for result in semantic_search(text, self._store, self.embedder, k=k):
            incident = self._incident_by_id(result.offset)

            if incident is not None:
                matches.append(
                    IncidentMatch(
                        incident=incident,
                        similarity=result.similarity,
                    )
                )

        return matches

    def search_similar_logs(
        self,
        error_text: str,
        k: int = 5,
    ) -> list[IncidentMatch]:
        """Return past incidents genuinely similar to ``error_text``.

        At most ``k`` results come back, and possibly none.  Ranking is
        left entirely to ``semantic_search``; this only drops matches too
        weak to be worth showing, so a fault with no real precedent
        returns an empty list instead of five bad guesses.
        """
        return [
            match
            for match in self._rank(error_text, k)
            if match.similarity >= self.min_similarity
        ][:k]

    # ------------------------------------------------------------------
    # Recording a new fault
    # ------------------------------------------------------------------

    def record_error(
        self,
        error_text: str,
        severity: str = DEFAULT_SEVERITY,
        subsystem: str | None = None,
        error_code: str | None = None,
        previous_action: str | None = None,
        outcome: str | None = None,
        k: int = 5,
    ) -> list[IncidentMatch]:
        """Match a new fault against history, then record it.

        Returns the Top ``k`` similar PREVIOUS incidents.  The new fault
        is stored only afterwards, so it is never matched against
        itself.  If it turns out to be a repeat of a known incident, that
        incident's count and ``last_seen`` are updated and NO new
        embedding is produced.
        """
        if severity not in SEVERITIES:
            raise ValueError(
                f"{severity} is not a fault severity; expected one of "
                f"{', '.join(SEVERITIES)}"
            )

        if not error_text.strip():
            raise ValueError("A fault must have a message")

        subsystem = normalize_subsystem(subsystem)

        # One scan serves both jobs: the shallow slice is displayed, the
        # deeper list is what deduplication reasons over.
        ranked = self._rank(error_text, max(k, DEDUP_CANDIDATE_COUNT))
        results = [
            match
            for match in ranked
            if match.similarity >= self.min_similarity
        ][:k]

        timestamp = self.now_fn()
        candidate = FaultIncident(
            incident_id=len(self.incidents) + 1,
            message=error_text,
            subsystem=subsystem,
            severity=severity,
            error_code=error_code,
            first_seen=timestamp,
            last_seen=timestamp,
            count=1,
            previous_action=previous_action,
            outcome=outcome,
            source_text=self.history.path.name,
            source_offset=len(self.incidents) + 1,
        )

        duplicate = find_duplicate(
            candidate,
            self.incidents,
            {
                match.incident.incident_id: match.similarity
                for match in ranked
            },
            self.dedup_threshold,
        )

        if duplicate is not None:
            # A repeat costs one metadata update and no embedding at all.
            duplicate.record_recurrence(timestamp)
            self.history.save(self.incidents)
            return results

        self.incidents.append(candidate)
        self.history.save(self.incidents)

        self._close_store()
        self._index([candidate])
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


def search_similar_logs(error_text: str, k: int = 5) -> list[IncidentMatch]:
    """Return past incidents most similar to ``error_text``."""
    return get_default_service().search_similar_logs(error_text, k=k)


def record_error(
    error_text: str,
    severity: str = DEFAULT_SEVERITY,
    subsystem: str | None = None,
    error_code: str | None = None,
    previous_action: str | None = None,
    outcome: str | None = None,
    k: int = 5,
) -> list[IncidentMatch]:
    """Match a new fault against history, then record it."""
    return get_default_service().record_error(
        error_text,
        severity=severity,
        subsystem=subsystem,
        error_code=error_code,
        previous_action=previous_action,
        outcome=outcome,
        k=k,
    )
