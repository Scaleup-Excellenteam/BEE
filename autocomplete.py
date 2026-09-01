"""Online autocomplete integration owned by Spec 3."""

from __future__ import annotations

from src.corpus.normalizer import normalize_text
from src.matching.matcher import calculate_best_match
from src.models import AutoCompleteData
from src.query_cache import (
    DEFAULT_QUERY_CACHE_CAPACITY,
    QueryCacheInfo,
    QueryResultCache,
)


_corpus_index = None
_RESULT_LIMIT = 5
_query_cache = QueryResultCache()


def set_corpus_index(
    index,
    *,
    query_cache_capacity: int = DEFAULT_QUERY_CACHE_CAPACITY,
) -> None:
    """Store an already initialized corpus index for reuse."""
    global _corpus_index, _query_cache
    _corpus_index = index
    _query_cache = QueryResultCache(query_cache_capacity)


def get_query_cache_info() -> QueryCacheInfo:
    """Return current autocomplete query-cache statistics."""
    return _query_cache.info()


def passes_partition_filter(query: str, sentence: str) -> bool:
    """Return whether a sentence can conservatively survive one query edit."""
    if len(query) < 2:
        return True

    for split in range(1, len(query)):
        left = query[:split]
        right = query[split:]

        if left not in sentence and right not in sentence:
            return False

    return True


def _build_result(candidate, score: int) -> AutoCompleteData:
    """Construct a completion while preserving candidate metadata."""
    return AutoCompleteData(
        completed_sentence=candidate.original_sentence,
        source_text=candidate.source_text,
        offset=candidate.offset,
        score=score,
    )


def _sort_and_limit(results: list[AutoCompleteData]) -> list[AutoCompleteData]:
    """Apply the shared score and alphabetical Top-5 contract."""
    results.sort(
        key=lambda result: (
            -result.score,
            result.completed_sentence,
        )
    )

    return results[:_RESULT_LIMIT]


def get_best_k_completions(prefix: str) -> list[AutoCompleteData]:
    """Return up to five best autocomplete results for a prefix."""
    if _corpus_index is None:
        raise RuntimeError("Corpus index has not been configured")

    query = normalize_text(prefix)
    if query == "":
        return []

    cached_completions = _query_cache.get(query)
    if cached_completions is not None:
        return cached_completions

    candidates = _corpus_index.get_candidates(query)
    exact_candidates = [
        candidate
        for candidate in candidates
        if query in candidate.normalized_sentence
    ]

    if len(exact_candidates) >= _RESULT_LIMIT:
        exact_score = 2 * len(query)
        exact_results = [
            _build_result(candidate, exact_score)
            for candidate in exact_candidates
        ]
        results = _sort_and_limit(exact_results)
        _query_cache.put(query, results)
        return results

    results = []

    for candidate in candidates:
        if not passes_partition_filter(
            query,
            candidate.normalized_sentence,
        ):
            continue

        score = calculate_best_match(
            query,
            candidate.normalized_sentence,
        )

        if score is None:
            continue

        results.append(_build_result(candidate, score))

    results = _sort_and_limit(results)
    _query_cache.put(query, results)
    return results
