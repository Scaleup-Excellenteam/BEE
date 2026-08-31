"""Semantic ranking over precomputed corpus embeddings."""

from src.semantic.contracts import EmbeddedSentence, SemanticResult
from src.semantic.similarity import cosine_similarity


def semantic_search(
    query: str,
    embedded_sentences: list[EmbeddedSentence],
    embedder,
    k: int = 5,
) -> list[SemanticResult]:
    """Return up to ``k`` corpus sentences ranked by semantic similarity."""
    if k <= 0 or not embedded_sentences or not query.strip():
        return []

    query_embedding = embedder(query)
    cosine_similarity(query_embedding, query_embedding)

    results = []
    for embedded_sentence in embedded_sentences:
        try:
            similarity = cosine_similarity(
                query_embedding,
                embedded_sentence.embedding,
            )
        except ValueError:
            continue

        results.append(
            SemanticResult(
                sentence=embedded_sentence.sentence,
                source_text=embedded_sentence.source_text,
                offset=embedded_sentence.offset,
                similarity=similarity,
            )
        )

    results.sort(
        key=lambda result: (
            -result.similarity,
            result.sentence,
        )
    )
    return results[:k]
