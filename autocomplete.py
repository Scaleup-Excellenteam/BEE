"""Online autocomplete integration owned by Spec 3."""

from __future__ import annotations

from src.corpus.normalizer import normalize_text
from src.matching.matcher import calculate_best_match
from src.models import AutoCompleteData


_corpus_index = None


def set_corpus_index(index) -> None:
    """Store an already initialized corpus index for reuse."""
    global _corpus_index
    _corpus_index = index


def get_best_k_completions(prefix: str) -> list[AutoCompleteData]:
    """Return up to five best autocomplete results for a prefix."""
    if _corpus_index is None:
        raise RuntimeError("Corpus index has not been configured")

    query = normalize_text(prefix)
    if query == "":
        return []

    candidates = _corpus_index.get_candidates(query)
    results = []

    for candidate in candidates:
        score = calculate_best_match(
            query,
            candidate.normalized_sentence,
        )

        if score is None:
            continue

        results.append(
            AutoCompleteData(
                completed_sentence=candidate.original_sentence,
                source_text=candidate.source_text,
                offset=candidate.offset,
                score=score,
            )
        )

    results.sort(
        key=lambda result: (
            -result.score,
            result.completed_sentence,
        )
    )

    return results[:5]
