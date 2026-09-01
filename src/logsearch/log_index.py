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

import os
import shutil
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
    RecordFaultResult,
    DEFAULT_SEVERITY,
    DEFAULT_SUBSYSTEM,
    FaultIncident,
    IncidentMatch,
    SEVERITIES,
    find_duplicate,
    normalize_subsystem,
)
from src.logsearch.retention import (
    CRITICAL,
    CRITICAL_RESERVE_RATIO,
    DEFAULT_MAX_INCIDENTS,
    DUPLICATE_UPDATED,
    NOT_STORED_CAPACITY,
    critical_reserved_slots,
    plan_admission,
    select_within_capacity,
)
from src.logsearch.log_parser import parse_fault_records


# Faults collected from previous satellites, previous missions and test
# environments.  Migrated into incidents once, on the first run.
BOOTSTRAP_DATASET_PATH = "data/historical_satellite_faults.log"

# The plain fault log written by earlier versions.  Still migrated on a
# first run so an existing deployment keeps its history, but no longer
# written to: incidents are the history now.
LEGACY_LOG_PATH = "logs/satellite_faults.log"

# The name the previous version exposed for the same file.  Kept as an
# alias so existing callers keep working; the file is now a legacy
# history that is migrated once rather than an active log.
DEFAULT_LOG_PATH = LEGACY_LOG_PATH

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
        max_incidents: int = DEFAULT_MAX_INCIDENTS,
        critical_reserve_ratio: float = CRITICAL_RESERVE_RATIO,
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
        self.max_incidents = max_incidents
        self.critical_reserve_ratio = critical_reserve_ratio
        self.warm_up_fn = warm_up_fn
        self.now_fn = now_fn or (
            lambda: datetime.now().strftime(TIMESTAMP_FORMAT)
        )

        self.incidents: list[FaultIncident] = self.history.load()
        self._by_id: dict[int, FaultIncident] = {}
        self._reindex()
        self._next_incident_id = self.history.next_incident_id(self.incidents)

        # Set when an upgrade is in progress: an older version's cache is
        # on disk, and a legacy history may still need converting.
        self.migrated_from_legacy = False
        self._legacy_cache = False

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

            if self._legacy_cache:
                # Written by an older version, so it cannot be searched.
                # It is retired during refresh, once the canonical
                # history it will be rebuilt from is safely on disk.
                self._close_store()

    def _legacy_cache_sources(self) -> set[str]:
        """Return cache source names an older version of this code wrote.

        Only the configured legacy log counts.  A cache belonging to some
        entirely different corpus is still refused.
        """
        if self.legacy_log_path is None:
            return set()

        return {self.legacy_log_path.name}

    def _verify_cache_matches_history(self) -> None:
        """Refuse a cache that does not belong to this history.

        Silently reusing the wrong cache would answer satellite queries
        out of some other corpus, which is worse than failing loudly.

        A cache this project wrote under its previous layout is the one
        exception: upgrading must not require the operator to delete
        files by hand, so it is flagged for rebuilding instead.
        """
        if self._store is None or len(self._store) == 0:
            return

        stored_source = self._store.metadata(0)["source_text"]

        if stored_source == CACHE_SOURCE_NAME:
            return

        if stored_source in self._legacy_cache_sources():
            self._legacy_cache = True
            return

        raise CacheError(
            f"The cache at {self.cache_path} was built from "
            f"{stored_source}, not {CACHE_SOURCE_NAME}; delete it or "
            "point at the right one"
        )

    def _discard_retired_cache(self) -> None:
        """Remove a cache retired during an upgrade.

        Only ever called once the replacement has been built AND
        validated, so until that point the old vectors stay on disk as a
        recovery option.
        """
        shutil.rmtree(f"{self.cache_path}.legacy", ignore_errors=True)

    def _retire_legacy_cache(self) -> None:
        """Set aside an older version's cache so a new one can be built.

        Called ONLY after the canonical incident history has been
        written, so the vectors being discarded are already reproducible.
        The directory is renamed rather than deleted, and removed only
        once the replacement has been built and validated.
        """
        self._close_store()

        retired = f"{self.cache_path}.legacy"
        shutil.rmtree(retired, ignore_errors=True)

        if os.path.isdir(self.cache_path):
            os.rename(self.cache_path, retired)

        self._legacy_cache = False

    def _cache_is_a_prefix(self) -> bool:
        """Return whether the cache still lines up with the history.

        Incident ``n`` must hold vector ``n``.  A cache that is longer
        than the history, or whose last vector belongs to a different
        incident, is stale -- normally because a compaction was
        interrupted.  It is not an error: the history is canonical, so
        the cache is simply rebuilt from it.
        """
        if self._store is None or len(self._store) == 0:
            return True

        indexed = len(self._store)

        if indexed > len(self.incidents):
            return False

        return (
            self._store.metadata(indexed - 1)["offset"]
            == self.incidents[indexed - 1].incident_id
        )

    def _validate_alignment(self) -> None:
        """Fail loudly if the cache and the history disagree in size."""
        indexed = self._indexed_count()

        if indexed != len(self.incidents):
            raise CacheError(
                f"The cache at {self.cache_path} holds {indexed} vectors "
                f"for {len(self.incidents)} incidents"
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

    def _incidents_from_records(self, records) -> list[FaultIncident]:
        """Convert parsed log records into structured incidents.

        Log lines carry no structure, so they convert with honest
        defaults: the parsed level becomes the severity, the parsed
        timestamp becomes both first and last seen, and the subsystem is
        UNKNOWN rather than guessed.  Nothing is invented -- no error
        codes, no actions, no outcomes.

        Repeats are folded together ONLY when the message and severity
        are character-for-character identical.  That is a fact about the
        text, not a similarity judgement, so it cannot merge two
        different faults; the semantic 0.90 rule is deliberately not used
        here, because a migration that quietly loses a distinct fault is
        unrecoverable.  A folded incident keeps every occurrence:

            count      how many times the line appears
            first_seen the earliest occurrence
            last_seen  the latest occurrence
        """
        incidents: dict[tuple[str, str], FaultIncident] = {}

        for record in records:
            key = (record.message, record.level)
            known = incidents.get(key)

            if known is None:
                incidents[key] = FaultIncident(
                    incident_id=len(incidents) + 1,
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
                continue

            known.count += 1
            known.first_seen = min(known.first_seen, record.timestamp)
            known.last_seen = max(known.last_seen, record.timestamp)

        return list(incidents.values())

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

        parsed = self._incidents_from_records(records)
        self.migrated_from_legacy = source == self.legacy_log_path

        # A prepared dataset larger than the configured budget is trimmed
        # by the same retention rules used at runtime, rather than
        # overflowing the memory it is supposed to fit in.
        self.incidents = select_within_capacity(
            parsed,
            self.max_incidents,
            self.critical_reserve_ratio,
        )

        # Ids come from the dataset order, so trimming leaves gaps and
        # the counter continues past every id ever issued.
        self._reindex()
        self._next_incident_id = len(parsed) + 1
        self.history.save(self.incidents, self._next_incident_id)

        return len(self.incidents)

    def _indexed_count(self) -> int:
        """Return how many incidents already have an embedding."""
        return len(self._store) if self._store is not None else 0

    def _index(
        self,
        incidents: list[FaultIncident],
        vectors: list[list[float]] | None = None,
    ) -> None:
        """Embed incidents and append them to the cache.

        ``vectors`` lets a caller hand over embeddings it has already
        computed, so nothing is embedded twice.  The caller is
        responsible for having closed the store first.
        """
        if not incidents:
            return

        precomputed = (
            None
            if vectors is None
            else dict(zip((one.incident_id for one in incidents), vectors))
        )
        writer = None

        try:
            for batch in batched(incidents, self.batch_size):
                if precomputed is None:
                    vectors = self.batch_embedder(
                        [incident.message for incident in batch]
                    )
                else:
                    vectors = [
                        precomputed[incident.incident_id]
                        for incident in batch
                    ]
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

        # Ordering matters.  The canonical history has just been written,
        # so every vector about to be discarded is now reproducible from
        # it.  Retiring the old cache any earlier would risk losing the
        # embeddings with nothing to rebuild them from.
        if self._legacy_cache:
            self._retire_legacy_cache()

        if not self._cache_is_a_prefix():
            return self._rebuild_cache()

        pending = self.incidents[self._indexed_count():]

        if not pending:
            return 0

        self._close_store()
        self._index(pending)
        self._open_store()
        self._validate_alignment()
        self._discard_retired_cache()

        return len(pending)

    def _rebuild_cache(self) -> int:
        """Throw the cache away and re-embed every incident.

        The history is the source of truth, so this always recovers, at
        the cost of embedding everything again.  It runs only when the
        cache cannot be trusted, never on a normal start.
        """
        self._close_store()
        shutil.rmtree(self.cache_path, ignore_errors=True)
        self._index(self.incidents)
        self._open_store()
        self._validate_alignment()

        self._discard_retired_cache()

        return len(self.incidents)

    def _compact_cache(
        self,
        previous_incidents: list[FaultIncident],
        final_incidents: list[FaultIncident],
        fresh_vectors: dict[int, list[float]],
    ) -> None:
        """Rewrite the cache to hold exactly the surviving incidents.

        Vectors that already exist are COPIED out of the current cache
        rather than recomputed, so evicting one incident costs no
        embeddings at all beyond the newcomer's own.  ``fresh_vectors``
        supplies the vectors the cache does not have yet.

        The new cache is built beside the old one and swapped in only
        once it is complete, so an interruption leaves either the old
        cache or a rebuildable gap, never a half-written index.
        """
        position_of = {
            incident.incident_id: index
            for index, incident in enumerate(previous_incidents)
        }
        existing = self._store.vectors() if self._store is not None else None

        def vector_for(incident: FaultIncident) -> list[float]:
            if incident.incident_id in fresh_vectors:
                return fresh_vectors[incident.incident_id]

            return existing[position_of[incident.incident_id]].tolist()

        staging = f"{self.cache_path}.rebuild"
        shutil.rmtree(staging, ignore_errors=True)

        writer = None

        try:
            for batch in batched(final_incidents, self.batch_size):
                items = [
                    EmbeddedSentence(
                        sentence=incident.message,
                        source_text=CACHE_SOURCE_NAME,
                        offset=incident.incident_id,
                        embedding=vector_for(incident),
                    )
                    for incident in batch
                ]

                if writer is None:
                    writer = EmbeddingStoreWriter(
                        staging,
                        dim=len(items[0].embedding),
                    )

                writer.append(items)

            if writer is not None:
                writer.finish()
        finally:
            if writer is not None:
                writer.close()

        # The memory map must be released before the directory moves.
        self._close_store()

        backup = f"{self.cache_path}.old"
        shutil.rmtree(backup, ignore_errors=True)

        if os.path.isdir(self.cache_path):
            os.rename(self.cache_path, backup)

        if os.path.isdir(staging):
            os.rename(staging, self.cache_path)

        shutil.rmtree(backup, ignore_errors=True)

        self._open_store()

    def warm_up(self) -> None:
        """Load the embedding model now rather than on the first query.

        This embeds one throwaway token, never any incident text, so
        calling it after ``refresh`` embeds nothing a second time.
        """
        self.warm_up_fn()

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def _reindex(self) -> None:
        """Rebuild the id lookup after the incident list changes.

        Ids are NOT positions.  Eviction leaves gaps -- a history of
        incidents 2 and 7 is perfectly normal -- so looking one up by
        subscript would return the wrong incident, or none at all.
        """
        self._by_id = {
            incident.incident_id: incident for incident in self.incidents
        }

    def _incident_by_id(self, incident_id: int) -> FaultIncident | None:
        """Return the incident with this id, if the history still has it."""
        return self._by_id.get(incident_id)

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
    ) -> RecordFaultResult:
        """Match a new fault against history, then try to record it.

        The result carries the Top ``k`` similar PREVIOUS incidents plus
        what became of this one in storage.  The search always happens
        first, so the fault is never matched against itself AND an
        operator still gets their answer even when the fault cannot be
        persisted.

        A repeat updates the known incident's count and ``last_seen`` and
        produces no embedding.  A genuinely new fault is stored if the
        retention policy allows, possibly displacing a less valuable
        incident; if it does not allow, the search result is still
        returned and ``stored`` is False.
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
            incident_id=self._next_incident_id,
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
            source_offset=self._next_incident_id,
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
            # A repeat costs one metadata update, no slot and no
            # embedding, so it is unaffected by capacity.
            duplicate.record_recurrence(timestamp)
            self.history.save(self.incidents, self._next_incident_id)

            return RecordFaultResult(
                matches=results,
                stored=True,
                deduplicated=True,
                incident_id=duplicate.incident_id,
                evicted_incident_id=None,
                reason=DUPLICATE_UPDATED,
            )

        plan = plan_admission(
            candidate,
            self.incidents,
            self.max_incidents,
            self.critical_reserve_ratio,
        )

        if not plan.store:
            # The fault was analysed and answered; it just is not worth a
            # slot compared with what memory already holds.
            return RecordFaultResult(
                matches=results,
                stored=False,
                deduplicated=False,
                incident_id=None,
                evicted_incident_id=None,
                reason=plan.reason,
            )

        vector = self.batch_embedder([candidate.message])[0]
        previous_incidents = list(self.incidents)
        self._next_incident_id = candidate.incident_id + 1

        if plan.evict is None:
            self.incidents.append(candidate)
            self._reindex()
            self.history.save(self.incidents, self._next_incident_id)

            self._close_store()
            self._index([candidate], vectors=[vector])
            self._open_store()
            evicted_id = None
        else:
            self.incidents = [
                incident
                for incident in previous_incidents
                if incident.incident_id != plan.evict.incident_id
            ] + [candidate]
            self._reindex()

            # Canonical history is written FIRST: if the cache rewrite
            # below is interrupted, the truth is already correct and the
            # cache is rebuilt on the next start.
            self.history.save(self.incidents, self._next_incident_id)
            self._compact_cache(
                previous_incidents,
                self.incidents,
                {candidate.incident_id: vector},
            )
            evicted_id = plan.evict.incident_id

        self._validate_alignment()

        return RecordFaultResult(
            matches=results,
            stored=True,
            deduplicated=False,
            incident_id=candidate.incident_id,
            evicted_incident_id=evicted_id,
            reason=plan.reason,
        )

    def storage_status(self) -> dict:
        """Return the configured semantic-memory budget and its usage.

        This is the memory budget the mission configured, not anything
        about the physical disk.
        """
        total = len(self.incidents)
        critical = sum(
            1 for incident in self.incidents if incident.severity == CRITICAL
        )

        return {
            "incidents": total,
            "max_incidents": self.max_incidents,
            "usage_percent": (
                round(total / self.max_incidents * 100, 1)
                if self.max_incidents
                else 0.0
            ),
            "critical_incidents": critical,
            "noncritical_incidents": total - critical,
            "critical_reserved_slots": critical_reserved_slots(
                self.max_incidents,
                self.critical_reserve_ratio,
            ),
        }


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
) -> RecordFaultResult:
    """Match a new fault against history, then try to record it."""
    return get_default_service().record_error(
        error_text,
        severity=severity,
        subsystem=subsystem,
        error_code=error_code,
        previous_action=previous_action,
        outcome=outcome,
        k=k,
    )
