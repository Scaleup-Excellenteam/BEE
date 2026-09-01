"""Local sentence embedding provider.

This is the offline counterpart of ``embedding_provider``.  It has the
same two public functions and the same shapes:

    embed_text(text)     one text  -> one vector
    embed_texts(texts)   many texts -> many vectors

but it runs a model on this machine instead of calling a web service.
There is no API key, no network request and no Gemini import anywhere in
this file, so the log search runtime never needs credentials.

The model is loaded lazily on first use and then kept.  Loading costs a
few seconds and a few hundred megabytes, so it must not happen at import
time: importing this module is free, which is what lets the tests run
with a stub and no model on disk.
"""

from __future__ import annotations


# A small, fast, widely used sentence embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vector width produced by that model.  Recorded here so a cache built
# with it can be recognised, and so tests can assert the expected shape.
EMBEDDING_DIMENSIONS = 384

# Texts handed to the model in one call.  The model runs locally, so this
# is a memory tradeoff rather than a quota one.
DEFAULT_BATCH_SIZE = 64


class LocalEmbeddingError(RuntimeError):
    """Raised when a text cannot be turned into an embedding vector."""


_model = None


def _load_model():
    """Return the sentence transformer model, loading it once."""
    global _model

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise LocalEmbeddingError(
            "sentence-transformers is not installed, so local embeddings "
            "are unavailable"
        ) from error

    try:
        _model = SentenceTransformer(MODEL_NAME)
    except Exception as error:
        raise LocalEmbeddingError(
            f"Could not load the local model {MODEL_NAME}: "
            f"{type(error).__name__}: {error}"
        ) from error

    return _model


def _encode(texts: list[str]) -> list[list[float]]:
    """Run the model over ``texts`` and return plain Python vectors."""
    model = _load_model()

    try:
        vectors = model.encode(texts)
    except Exception as error:
        raise LocalEmbeddingError(
            f"Local embedding failed: {type(error).__name__}: {error}"
        ) from error

    return [[float(value) for value in vector] for vector in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text, in the same order."""
    texts = list(texts)

    if not texts:
        return []

    vectors = _encode(texts)

    if len(vectors) != len(texts):
        raise LocalEmbeddingError(
            f"The model returned {len(vectors)} embeddings for "
            f"{len(texts)} texts"
        )

    for position, vector in enumerate(vectors):
        if not vector:
            raise LocalEmbeddingError(f"Embedding {position} is empty")

    return vectors


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for one complete text."""
    return embed_texts([text])[0]
