"""Tests for semantic search over precomputed sentence embeddings."""

from unittest.mock import Mock

import pytest

from src.semantic.contracts import EmbeddedSentence, SemanticResult
from src.semantic.search import semantic_search


def embedded_sentence(
    sentence: str,
    embedding: list[float],
    source_text: str = "source.txt",
    offset: int = 1,
) -> EmbeddedSentence:
    return EmbeddedSentence(
        sentence=sentence,
        source_text=source_text,
        offset=offset,
        embedding=embedding,
    )


def test_query_is_embedded_once_and_results_rank_descending():
    embedder = Mock(return_value=[1.0, 0.0])
    corpus = [
        embedded_sentence("Least similar", [0.0, 1.0]),
        embedded_sentence("Most similar", [1.0, 0.0]),
        embedded_sentence("Middle", [1.0, 1.0]),
    ]

    results = semantic_search("meaningful query", corpus, embedder)

    embedder.assert_called_once_with("meaningful query")
    assert [result.sentence for result in results] == [
        "Most similar",
        "Middle",
        "Least similar",
    ]
    assert [result.similarity for result in results] == sorted(
        (result.similarity for result in results),
        reverse=True,
    )


def test_equal_similarity_uses_alphabetical_sentence_tie_breaker():
    corpus = [
        embedded_sentence("Zulu", [1.0, 0.0]),
        embedded_sentence("Alpha", [1.0, 0.0]),
        embedded_sentence("Beta", [1.0, 0.0]),
    ]

    results = semantic_search("query", corpus, lambda _: [1.0, 0.0])

    assert [result.sentence for result in results] == [
        "Alpha",
        "Beta",
        "Zulu",
    ]


def test_default_top_five_limit():
    corpus = [
        embedded_sentence(f"Sentence {index}", [1.0, float(index)])
        for index in range(6)
    ]

    results = semantic_search("query", corpus, lambda _: [1.0, 0.0])

    assert len(results) == 5


def test_custom_k_limit():
    corpus = [
        embedded_sentence(f"Sentence {index}", [1.0, float(index)])
        for index in range(4)
    ]

    results = semantic_search(
        "query",
        corpus,
        lambda _: [1.0, 0.0],
        k=2,
    )

    assert len(results) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty_without_embedding_query(k):
    embedder = Mock()

    results = semantic_search(
        "query",
        [embedded_sentence("Sentence", [1.0, 0.0])],
        embedder,
        k=k,
    )

    assert results == []
    embedder.assert_not_called()


def test_fewer_than_k_returns_every_valid_result():
    corpus = [
        embedded_sentence("One", [1.0, 0.0]),
        embedded_sentence("Two", [0.0, 1.0]),
    ]

    results = semantic_search(
        "query",
        corpus,
        lambda _: [1.0, 0.0],
        k=10,
    )

    assert len(results) == 2


def test_empty_corpus_returns_empty_without_embedding_query():
    embedder = Mock()

    assert semantic_search("query", [], embedder) == []
    embedder.assert_not_called()


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_empty_or_whitespace_query_returns_empty_without_embedding(query):
    embedder = Mock()

    results = semantic_search(
        query,
        [embedded_sentence("Sentence", [1.0, 0.0])],
        embedder,
    )

    assert results == []
    embedder.assert_not_called()


def test_result_preserves_sentence_source_offset_and_similarity():
    corpus = [
        embedded_sentence(
            sentence="Original corpus sentence",
            embedding=[1.0, 0.0],
            source_text="docs/a.txt",
            offset=42,
        )
    ]

    results = semantic_search("query", corpus, lambda _: [1.0, 0.0])

    assert results == [
        SemanticResult(
            sentence="Original corpus sentence",
            source_text="docs/a.txt",
            offset=42,
            similarity=pytest.approx(1.0),
        )
    ]


@pytest.mark.parametrize(
    "invalid_embedding",
    [
        [],
        [1.0],
        [0.0, 0.0],
    ],
)
def test_invalid_individual_corpus_embedding_is_skipped(invalid_embedding):
    corpus = [
        embedded_sentence("Invalid", invalid_embedding),
        embedded_sentence("Valid", [1.0, 0.0]),
    ]

    results = semantic_search("query", corpus, lambda _: [1.0, 0.0])

    assert [result.sentence for result in results] == ["Valid"]


@pytest.mark.parametrize("invalid_query_embedding", [[], [0.0, 0.0]])
def test_invalid_query_embedding_raises_value_error(invalid_query_embedding):
    corpus = [embedded_sentence("Sentence", [1.0, 0.0])]

    with pytest.raises(ValueError):
        semantic_search(
            "query",
            corpus,
            lambda _: invalid_query_embedding,
        )


def test_embedder_exception_propagates():
    corpus = [embedded_sentence("Sentence", [1.0, 0.0])]

    def failing_embedder(_):
        raise RuntimeError("embedding service unavailable")

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        semantic_search("query", corpus, failing_embedder)
