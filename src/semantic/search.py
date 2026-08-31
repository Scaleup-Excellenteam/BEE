"""Semantic ranking over precomputed corpus embeddings."""

import heapq
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

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


def _query_vector(query: str, embedder) -> np.ndarray:
    """Embed and validate one query as a float32 NumPy vector."""
    embedding = embedder(query)

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "query embedding must contain only numeric values"
        ) from error

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("query embedding must be a non-empty vector")

    if not np.all(np.isfinite(vector)):
        raise ValueError("query embedding must contain only finite values")

    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("query embedding must have non-zero magnitude")

    return vector


def semantic_search_store(
    query: str,
    store,
    embedder,
    k: int = 5,
) -> list[SemanticResult]:
    """Search a memory-mapped embedding store in vectorized chunks."""
    if k <= 0 or not query.strip() or len(store) == 0:
        return []

    query_vector = _query_vector(query, embedder)
    if query_vector.size != store.dim:
        raise ValueError(
            "query embedding dimensions do not match the embedding store"
        )

    query_magnitude = float(np.linalg.norm(query_vector))
    top_candidates: list[tuple[float, int]] = []

    for start_index, block in store.iter_blocks():
        vectors = np.asarray(block, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != query_vector.size:
            raise ValueError("embedding store yielded an invalid block")

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            magnitudes = np.linalg.norm(vectors, axis=1)
            similarities = (
                vectors @ query_vector
            ) / (magnitudes * query_magnitude)

        valid_positions = np.flatnonzero(
            np.isfinite(similarities) & (magnitudes > 0.0)
        )

        for relative_index in valid_positions:
            absolute_index = start_index + int(relative_index)
            candidate = (
                float(similarities[relative_index]),
                -absolute_index,
            )

            if len(top_candidates) < k:
                heapq.heappush(top_candidates, candidate)
            elif candidate > top_candidates[0]:
                heapq.heapreplace(top_candidates, candidate)

    results = []
    for similarity, negative_index in top_candidates:
        metadata = store.metadata(-negative_index)
        results.append(
            SemanticResult(
                sentence=metadata["sentence"],
                source_text=metadata["source_text"],
                offset=metadata["offset"],
                similarity=similarity,
            )
        )

    results.sort(
        key=lambda result: (
            -result.similarity,
            result.sentence,
        )
    )
    return results
