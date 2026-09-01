"""Gemini embedding provider (Spec 1).

This module turns ONE piece of text into ONE embedding vector.

    "Authentication is required."  ->  [0.12, -0.43, 0.81, ...]

The whole text is sent to Gemini as a single string.  This has nothing to
do with the Part A trigram index: trigrams are a lexical retrieval trick,
while an embedding is a semantic vector produced by the model.

There are two entry points:

    embed_text(text)     one text  -> one vector.  Used for a live query.
    embed_texts(texts)   many texts -> many vectors, in ONE request.

``embed_texts`` is what makes building a large corpus affordable: sending
100 sentences per request turns 2.58 million requests into about 26,000.

Every failure -- a missing key, a dead network, a timeout, a response that
does not look the way we expect -- is converted into ``EmbeddingError``.
Nothing raw from the SDK is allowed to escape, so a semantic failure can
never take down the Part A autocomplete path.
"""

from __future__ import annotations

import os


# Gemini's current text embedding model.
EMBEDDING_MODEL = "gemini-embedding-001"

# The model defaults to 3072 numbers per vector.  We ask for 768 instead:
# it is four times smaller on disk and in RAM, and cosine similarity is
# unaffected by the choice, so Developer 2's ranking is not impacted.
EMBEDDING_DIMENSIONS = 768

# The API key is read from here, every call, and never stored in a module
# level constant.
API_KEY_ENVIRONMENT_VARIABLE = "GEMINI_API_KEY"

# Milliseconds to wait for one embedding request before giving up.  A batch
# request carries more work than a single one, so this is generous.
REQUEST_TIMEOUT_MILLISECONDS = 120_000

# How many texts to put in one batch request.  Gemini caps the number of
# texts per embed_content call, so callers should not raise this blindly.
MAX_BATCH_SIZE = 100


class EmbeddingError(RuntimeError):
    """Raised when a text cannot be turned into an embedding vector."""


def _read_api_key() -> str:
    """Return the Gemini API key from the environment."""
    api_key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE, "").strip()

    if not api_key:
        raise EmbeddingError(
            "Missing Gemini API key: set the "
            f"{API_KEY_ENVIRONMENT_VARIABLE} environment variable"
        )

    return api_key


def _hide_api_key(message: str, api_key: str) -> str:
    """Return ``message`` with the API key blanked out.

    Third party error text can quote the request that failed, so it is
    scrubbed before it reaches a log file or a terminal.
    """
    return message.replace(api_key, "***")


def _create_client(api_key: str):
    """Build the Gemini client.

    Kept as its own function so tests can replace it and never open a
    network connection.
    """
    from google import genai

    return genai.Client(api_key=api_key)


def _request_embedding(client, text: str):
    """Send one complete text to Gemini and return the raw response."""
    from google.genai import types

    return client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MILLISECONDS,
            ),
        ),
    )


def _extract_embedding(response) -> list[float]:
    """Pull the vector out of a Gemini response, or fail loudly."""
    embeddings = getattr(response, "embeddings", None)

    if not embeddings:
        raise EmbeddingError("Gemini response contained no embeddings")

    values = getattr(embeddings[0], "values", None)

    if values is None:
        raise EmbeddingError("Gemini response contained no embedding values")

    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise EmbeddingError(
            "Gemini returned an embedding that is not a list of numbers"
        ) from error

    if not vector:
        raise EmbeddingError("Gemini returned an empty embedding")

    return vector


def embed_text(text: str) -> list[float]:
    """Return the Gemini embedding vector for one complete text."""
    api_key = _read_api_key()

    try:
        client = _create_client(api_key)
        response = _request_embedding(client, text)
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(
            "Gemini embedding request failed: "
            + _hide_api_key(f"{type(error).__name__}: {error}", api_key)
        ) from error

    return _extract_embedding(response)


def _request_embeddings(client, texts: list[str]):
    """Send several complete texts to Gemini in one request."""
    from google.genai import types

    return client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=list(texts),
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MILLISECONDS,
            ),
        ),
    )


def _extract_embeddings(response, expected_count: int) -> list[list[float]]:
    """Pull one vector per input text out of a batch response."""
    embeddings = getattr(response, "embeddings", None)

    if not embeddings:
        raise EmbeddingError("Gemini response contained no embeddings")

    if len(embeddings) != expected_count:
        raise EmbeddingError(
            f"Gemini returned {len(embeddings)} embeddings for "
            f"{expected_count} texts"
        )

    return [
        _extract_embedding_values(embedding, position)
        for position, embedding in enumerate(embeddings)
    ]


def _extract_embedding_values(embedding, position: int) -> list[float]:
    """Validate and convert one embedding out of a batch response."""
    values = getattr(embedding, "values", None)

    if values is None:
        raise EmbeddingError(
            f"Gemini embedding {position} contained no values"
        )

    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise EmbeddingError(
            f"Gemini embedding {position} is not a list of numbers"
        ) from error

    if not vector:
        raise EmbeddingError(f"Gemini embedding {position} is empty")

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text, using a single request.

    The result is in the same order as ``texts``, so the caller can pair
    them up by position.
    """
    texts = list(texts)

    if not texts:
        return []

    if len(texts) > MAX_BATCH_SIZE:
        raise EmbeddingError(
            f"Batch of {len(texts)} texts exceeds the maximum of "
            f"{MAX_BATCH_SIZE}"
        )

    api_key = _read_api_key()

    try:
        client = _create_client(api_key)
        response = _request_embeddings(client, texts)
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(
            "Gemini batch embedding request failed: "
            + _hide_api_key(f"{type(error).__name__}: {error}", api_key)
        ) from error

    return _extract_embeddings(response, len(texts))
