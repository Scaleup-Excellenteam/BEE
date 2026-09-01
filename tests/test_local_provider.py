"""Tests for the local sentence embedding provider.

No test loads or downloads the real model: the model object is replaced
with a stub, so these run offline and in milliseconds.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.semantic import local_provider
from src.semantic.local_provider import (
    LocalEmbeddingError,
    embed_text,
    embed_texts,
)


@pytest.fixture(autouse=True)
def forget_loaded_model(monkeypatch):
    """Give every test a provider with no model loaded yet."""
    monkeypatch.setattr(local_provider, "_model", None)


def stub_model(vectors):
    """Return an object shaped like a SentenceTransformer."""
    return SimpleNamespace(encode=Mock(return_value=vectors))


def test_embeds_a_batch_of_texts(monkeypatch):
    monkeypatch.setattr(
        local_provider,
        "_model",
        stub_model([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert embed_texts(["first", "second"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embeds_a_single_text(monkeypatch):
    monkeypatch.setattr(local_provider, "_model", stub_model([[1.0, 2.0]]))

    assert embed_text("only") == [1.0, 2.0]


def test_sends_the_complete_text_to_the_model(monkeypatch):
    model = stub_model([[1.0]])
    monkeypatch.setattr(local_provider, "_model", model)

    embed_text("Authentication is required.")

    model.encode.assert_called_once_with(["Authentication is required."])


def test_numpy_style_vectors_become_plain_floats(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        local_provider,
        "_model",
        stub_model(np.array([[1.5, 2.5]], dtype=np.float32)),
    )

    vector = embed_text("text")

    assert vector == [1.5, 2.5]
    assert all(isinstance(value, float) for value in vector)


def test_no_texts_means_no_model_call(monkeypatch):
    model = stub_model([])
    monkeypatch.setattr(local_provider, "_model", model)

    assert embed_texts([]) == []
    model.encode.assert_not_called()


def test_the_model_is_loaded_once_and_reused(monkeypatch):
    created = Mock(return_value=stub_model([[1.0]]))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=created),
    )

    embed_text("first")
    embed_text("second")

    created.assert_called_once_with(local_provider.MODEL_NAME)


def test_a_missing_library_is_reported_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(LocalEmbeddingError, match="not installed"):
        embed_text("text")


def test_a_model_that_will_not_load_is_reported_clearly(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=Mock(side_effect=OSError("no such model"))
        ),
    )

    with pytest.raises(LocalEmbeddingError, match="Could not load"):
        embed_text("text")


def test_a_failing_model_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(
        local_provider,
        "_model",
        SimpleNamespace(encode=Mock(side_effect=RuntimeError("out of memory"))),
    )

    with pytest.raises(LocalEmbeddingError, match="Local embedding failed"):
        embed_text("text")


def test_a_wrong_number_of_vectors_is_rejected(monkeypatch):
    monkeypatch.setattr(local_provider, "_model", stub_model([[1.0]]))

    with pytest.raises(LocalEmbeddingError, match="1 embeddings for 2"):
        embed_texts(["first", "second"])


def test_an_empty_vector_is_rejected(monkeypatch):
    monkeypatch.setattr(local_provider, "_model", stub_model([[1.0], []]))

    with pytest.raises(LocalEmbeddingError, match="Embedding 1 is empty"):
        embed_texts(["first", "second"])


def test_no_api_key_is_needed(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(local_provider, "_model", stub_model([[1.0, 2.0]]))

    assert embed_text("text") == [1.0, 2.0]
