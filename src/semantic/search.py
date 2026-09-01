"""Semantic ranking over precomputed corpus embeddings."""

import heapq
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from src.semantic.contracts import EmbeddedSentence, SemanticResult
from src.semantic.similarity import cosine_similarity


@dataclass(slots=True)
class _RankedCandidate:
    """Heap entry ordered from the worst retained candidate upward."""

    embedded_sentence: EmbeddedSentence
    similarity: float
    position: int

    def __lt__(self, other: "_RankedCandidate") -> bool:
        if self.similarity != other.similarity:
            return self.similarity < other.similarity

        if self.embedded_sentence.sentence != other.embedded_sentence.sentence:
            return (
                self.embedded_sentence.sentence
                > other.embedded_sentence.sentence
            )

        return self.position > other.position


def semantic_search(
    query: str,
    embedded_sentences: Iterable[EmbeddedSentence],
    embedder,
    k: int = 5,
) -> list[SemanticResult]:
    """Return up to ``k`` corpus sentences ranked by semantic similarity."""
    if k <= 0 or not query.strip():
        return []

    sentence_iterator = iter(embedded_sentences)
    try:
        first_sentence = next(sentence_iterator)
    except StopIteration:
        return []

    query_embedding = embedder(query)
    cosine_similarity(query_embedding, query_embedding)

    top_candidates: list[_RankedCandidate] = []
    for position, embedded_sentence in enumerate(
        itertools.chain((first_sentence,), sentence_iterator)
    ):
        try:
            similarity = cosine_similarity(
                query_embedding,
                embedded_sentence.embedding,
            )
        except ValueError:
            continue

        candidate = _RankedCandidate(
            embedded_sentence=embedded_sentence,
            similarity=similarity,
            position=position,
        )

        if len(top_candidates) < k:
            heapq.heappush(top_candidates, candidate)
        elif top_candidates[0] < candidate:
            heapq.heapreplace(top_candidates, candidate)

    top_candidates.sort(
        key=lambda candidate: (
            -candidate.similarity,
            candidate.embedded_sentence.sentence,
            candidate.position,
        )
    )

    return [
        SemanticResult(
            sentence=candidate.embedded_sentence.sentence,
            source_text=candidate.embedded_sentence.source_text,
            offset=candidate.embedded_sentence.offset,
            similarity=candidate.similarity,
        )
        for candidate in top_candidates
    ]
