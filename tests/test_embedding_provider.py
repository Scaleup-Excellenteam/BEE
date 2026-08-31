"""Tests for the Gemini embedding provider.

Every test runs completely offline: ``_create_client`` and
``_request_embedding`` are replaced, so no real API call is ever made.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.semantic import embedding_provider
from src.semantic.embedding_provider import EmbeddingError, embed_text


API_KEY_VARIABLE = embedding_provider.API_KEY_ENVIRONMENT_VARIABLE


def fake_response(values):
    """Build an object shaped like a Gemini embedding response."""
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


@pytest.fixture
def api_key(monkeypatch):
    """Provide a fake API key and a fake client for one test."""
    monkeypatch.setenv(API_KEY_VARIABLE, "test-api-key")
    monkeypatch.setattr(
        embedding_provider,
        "_create_client",
        Mock(return_value=SimpleNamespace()),
    )

    return "test-api-key"


def test_returns_embedding_vector_for_text(api_key, monkeypatch):
    request = Mock(return_value=fake_response([0.12, -0.43, 0.81]))
    monkeypatch.setattr(embedding_provider, "_request_embedding", request)

    assert embed_text("Authentication is required.") == [0.12, -0.43, 0.81]


def test_sends_the_complete_text_without_splitting_it(api_key, monkeypatch):
    request = Mock(return_value=fake_response([1.0]))
    monkeypatch.setattr(embedding_provider, "_request_embedding", request)

    embed_text("Authentication is required.")

    _, sent_text = request.call_args.args
    assert sent_text == "Authentication is required."


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    create_client = Mock()
    monkeypatch.setattr(embedding_provider, "_create_client", create_client)

    with pytest.raises(EmbeddingError, match="Missing Gemini API key"):
        embed_text("anything")

    create_client.assert_not_called()


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv(API_KEY_VARIABLE, "   ")

    with pytest.raises(EmbeddingError, match="Missing Gemini API key"):
        embed_text("anything")


def test_api_failure_is_wrapped_in_embedding_error(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(side_effect=ConnectionError("network is down")),
    )

    with pytest.raises(EmbeddingError, match="request failed") as failure:
        embed_text("anything")

    assert isinstance(failure.value.__cause__, ConnectionError)


def test_timeout_is_wrapped_in_embedding_error(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(side_effect=TimeoutError("deadline exceeded")),
    )

    with pytest.raises(EmbeddingError, match="request failed"):
        embed_text("anything")


def test_api_key_is_never_exposed_in_the_error_message(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(side_effect=RuntimeError(f"call failed with key={api_key}")),
    )

    with pytest.raises(EmbeddingError) as failure:
        embed_text("anything")

    assert api_key not in str(failure.value)
    assert "***" in str(failure.value)


def test_response_without_embeddings_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(return_value=SimpleNamespace(embeddings=None)),
    )

    with pytest.raises(EmbeddingError, match="no embeddings"):
        embed_text("anything")


def test_invalid_response_shape_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(return_value=SimpleNamespace(unexpected="shape")),
    )

    with pytest.raises(EmbeddingError, match="no embeddings"):
        embed_text("anything")


def test_missing_embedding_values_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(return_value=fake_response(None)),
    )

    with pytest.raises(EmbeddingError, match="no embedding values"):
        embed_text("anything")


def test_non_numeric_embedding_values_raise(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(return_value=fake_response(["not", "numbers"])),
    )

    with pytest.raises(EmbeddingError, match="not a list of numbers"):
        embed_text("anything")


def test_empty_embedding_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embedding",
        Mock(return_value=fake_response([])),
    )

    with pytest.raises(EmbeddingError, match="empty embedding"):
        embed_text("anything")


def batch_response(vectors):
    """Build an object shaped like a batched Gemini response."""
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=values) for values in vectors]
    )


def test_batch_returns_one_vector_per_text(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embeddings",
        Mock(return_value=batch_response([[1.0, 2.0], [3.0, 4.0]])),
    )

    assert embedding_provider.embed_texts(["first", "second"]) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]


def test_batch_sends_all_texts_in_one_request(api_key, monkeypatch):
    request = Mock(return_value=batch_response([[1.0], [2.0], [3.0]]))
    monkeypatch.setattr(embedding_provider, "_request_embeddings", request)

    embedding_provider.embed_texts(["a", "b", "c"])

    assert request.call_count == 1
    _, sent_texts = request.call_args.args
    assert sent_texts == ["a", "b", "c"]


def test_batch_of_no_texts_makes_no_request(api_key, monkeypatch):
    request = Mock()
    monkeypatch.setattr(embedding_provider, "_request_embeddings", request)

    assert embedding_provider.embed_texts([]) == []
    request.assert_not_called()


def test_oversized_batch_is_rejected_before_any_request(api_key, monkeypatch):
    request = Mock()
    monkeypatch.setattr(embedding_provider, "_request_embeddings", request)
    texts = ["x"] * (embedding_provider.MAX_BATCH_SIZE + 1)

    with pytest.raises(EmbeddingError, match="exceeds the maximum"):
        embedding_provider.embed_texts(texts)

    request.assert_not_called()


def test_batch_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)

    with pytest.raises(EmbeddingError, match="Missing Gemini API key"):
        embedding_provider.embed_texts(["a"])


def test_batch_api_failure_is_wrapped(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embeddings",
        Mock(side_effect=ConnectionError("network is down")),
    )

    with pytest.raises(EmbeddingError, match="batch embedding request failed"):
        embedding_provider.embed_texts(["a"])


def test_batch_with_wrong_number_of_embeddings_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embeddings",
        Mock(return_value=batch_response([[1.0]])),
    )

    with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2"):
        embedding_provider.embed_texts(["a", "b"])


def test_batch_with_an_empty_embedding_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embeddings",
        Mock(return_value=batch_response([[1.0], []])),
    )

    with pytest.raises(EmbeddingError, match="embedding 1 is empty"):
        embedding_provider.embed_texts(["a", "b"])


def test_batch_with_non_numeric_values_raises(api_key, monkeypatch):
    monkeypatch.setattr(
        embedding_provider,
        "_request_embeddings",
        Mock(return_value=batch_response([["nope"]])),
    )

    with pytest.raises(EmbeddingError, match="embedding 0 is not a list"):
        embedding_provider.embed_texts(["a"])
