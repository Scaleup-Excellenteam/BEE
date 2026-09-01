"""Tests for the bounded in-memory autocomplete query cache."""

import pytest

from src.models import AutoCompleteData
from src.query_cache import (
    DEFAULT_QUERY_CACHE_CAPACITY,
    QueryResultCache,
)


def _result(sentence: str) -> AutoCompleteData:
    """Build a compact autocomplete result for cache tests."""
    return AutoCompleteData(sentence, "source.txt", 0, 10)


def test_default_capacity_is_five_hundred() -> None:
    cache = QueryResultCache()

    assert cache.capacity == DEFAULT_QUERY_CACHE_CAPACITY == 500


def test_cache_records_miss_then_hit_including_empty_results() -> None:
    cache = QueryResultCache()

    assert cache.get("missing") is None
    cache.put("missing", [])
    assert cache.get("missing") == []

    info = cache.info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.hit_rate == 0.5


def test_cache_eviction_preserves_more_frequent_query() -> None:
    cache = QueryResultCache(capacity=2)
    cache.put("frequent", [_result("Frequent")])
    cache.put("old-and-rare", [_result("Rare")])
    assert cache.get("frequent") is not None

    cache.put("new", [_result("New")])

    assert cache.get("frequent") is not None
    assert cache.get("old-and-rare") is None
    assert cache.get("new") is not None
    assert cache.info().evictions == 1


def test_cache_breaks_frequency_ties_by_oldest_access() -> None:
    cache = QueryResultCache(capacity=2)
    cache.put("oldest", [_result("Old")])
    cache.put("newer", [_result("Newer")])

    cache.put("newest", [_result("Newest")])

    assert cache.get("oldest") is None
    assert cache.get("newer") is not None
    assert cache.get("newest") is not None


def test_cache_returns_defensive_copies() -> None:
    cache = QueryResultCache()
    original = [_result("Original")]
    cache.put("query", original)

    original[0].completed_sentence = "Changed outside"
    first_hit = cache.get("query")
    assert first_hit is not None
    first_hit[0].completed_sentence = "Changed hit"

    second_hit = cache.get("query")
    assert second_hit is not None
    assert second_hit[0].completed_sentence == "Original"


def test_cache_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        QueryResultCache(capacity=0)
