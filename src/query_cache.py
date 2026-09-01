"""Bounded in-memory LFU cache for autocomplete query results."""

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock

from src.models import AutoCompleteData


DEFAULT_QUERY_CACHE_CAPACITY = 500


@dataclass(frozen=True, slots=True)
class QueryCacheInfo:
    """Expose a read-only snapshot of query-cache state."""

    capacity: int
    size: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        """Return the fraction of lookups served from cache.

        Returns:
            A value between zero and one, or zero before any lookup.
        """
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


@dataclass(slots=True)
class _CacheEntry:
    """Store one cached result and its LFU eviction metadata."""

    results: tuple[AutoCompleteData, ...]
    frequency: int
    last_access: int


class QueryResultCache:
    """Keep the most frequently reused autocomplete results in memory."""

    def __init__(self, capacity: int = DEFAULT_QUERY_CACHE_CAPACITY) -> None:
        """Initialize an empty bounded cache.

        Args:
            capacity: Maximum number of normalized queries to retain.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError("query cache capacity must be positive")

        self._capacity = capacity
        self._entries: dict[str, _CacheEntry] = {}
        self._clock = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    @property
    def capacity(self) -> int:
        """Return the maximum number of cached queries.

        Returns:
            The configured query capacity.
        """
        return self._capacity

    def __len__(self) -> int:
        """Return the number of currently cached queries.

        Returns:
            The number of resident entries.
        """
        with self._lock:
            return len(self._entries)

    def get(self, query: str) -> list[AutoCompleteData] | None:
        """Look up a normalized query and record a hit or miss.

        Args:
            query: Normalized English query used by the completion engine.

        Returns:
            A defensive copy of cached results, or ``None`` on a miss.
        """
        with self._lock:
            entry = self._entries.get(query)
            if entry is None:
                self._misses += 1
                return None

            self._clock += 1
            self._hits += 1
            entry.frequency += 1
            entry.last_access = self._clock
            return list(deepcopy(entry.results))

    def put(self, query: str, results: list[AutoCompleteData]) -> None:
        """Store results and evict the least-frequently-used entry if needed.

        The oldest access breaks ties between entries with equal frequency.

        Args:
            query: Normalized English query used by the completion engine.
            results: Up to five autocomplete results to cache.
        """
        with self._lock:
            self._clock += 1
            existing = self._entries.get(query)
            if existing is not None:
                existing.results = tuple(deepcopy(results))
                existing.frequency += 1
                existing.last_access = self._clock
                return

            if len(self._entries) >= self._capacity:
                evicted_query = min(
                    self._entries,
                    key=lambda key: (
                        self._entries[key].frequency,
                        self._entries[key].last_access,
                    ),
                )
                del self._entries[evicted_query]
                self._evictions += 1

            self._entries[query] = _CacheEntry(
                results=tuple(deepcopy(results)),
                frequency=1,
                last_access=self._clock,
            )

    def clear(self) -> None:
        """Remove all entries and reset cache statistics."""
        with self._lock:
            self._entries.clear()
            self._clock = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def info(self) -> QueryCacheInfo:
        """Return a consistent snapshot of cache statistics.

        Returns:
            Current capacity, occupancy, and lookup counters.
        """
        with self._lock:
            return QueryCacheInfo(
                capacity=self._capacity,
                size=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
