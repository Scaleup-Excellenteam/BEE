"""Local sentence embedding provider.

This is the offline counterpart of ``embedding_provider``.  It has the
same two public functions and the same shapes:

    embed_text(text)     one text  -> one vector
    embed_texts(texts)   many texts -> many vectors

but it runs a model on this machine instead of calling a web service.
Inference is local: there is no API key, no cloud inference service and
no Gemini import anywhere in this file, so the log search runtime never
needs credentials.

The model is loaded lazily on first use and then kept.  Loading costs a
few seconds and a few hundred megabytes, so it must not happen at import
time: importing this module is free, which is what lets the tests run
with a stub and no model on disk.  Call ``warm_up()`` during startup to
pay that cost somewhere predictable instead of on the first query.

Loading prefers the local Hugging Face cache.  Once the model is
installed locally, runtime search does not require a cloud API and needs
no network access.  A machine that does not have the model yet will
download it once, so an air gapped deployment must have the model
preinstalled.
"""

from __future__ import annotations

from src.logging_config import get_application_logger


LOGGER = get_application_logger()

# A small, fast, widely used sentence embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vector width produced by that model.  Recorded here so a cache built
# with it can be recognised, and so tests can assert the expected shape.
EMBEDDING_DIMENSIONS = 384

# Texts handed to the model in one call.  The model runs locally, so this
# is a memory tradeoff rather than a quota one.
DEFAULT_BATCH_SIZE = 64

# Throwaway text used only to force the model to load.
WARM_UP_TEXT = "warm up"


class LocalEmbeddingError(RuntimeError):
    """Raised when a text cannot be turned into an embedding vector."""


_model = None


def _import_sentence_transformer():
    """Return the SentenceTransformer class, or explain its absence."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise LocalEmbeddingError(
            "sentence-transformers is not installed, so local embeddings "
            "are unavailable"
        ) from error

    return SentenceTransformer


def _load_model():
    """Return the sentence transformer model, loading it once.

    The local copy is tried FIRST, so a machine that already has the
    model in its Hugging Face cache loads it without contacting the Hub,
    avoiding the "unauthenticated request" round trip on every start.
    This is what an air gapped deployment relies on, and it requires the
    model to have been installed there beforehand.

    Only if there is no local copy does it fall back to a download, so a
    fresh development machine can still bootstrap itself.  No token is
    ever required or sent.
    """
    global _model

    if _model is not None:
        return _model

    sentence_transformer = _import_sentence_transformer()

    try:
        _model = sentence_transformer(MODEL_NAME, local_files_only=True)
        return _model
    except Exception:
        LOGGER.info(
            "The local model %s was not found in the local cache, "
            "downloading it once; an air gapped deployment needs it "
            "preinstalled.",
            MODEL_NAME,
        )

    try:
        _model = sentence_transformer(MODEL_NAME)
    except Exception as error:
        raise LocalEmbeddingError(
            f"Could not load the local model {MODEL_NAME}: "
            f"{type(error).__name__}: {error}"
        ) from error

    return _model


def warm_up() -> None:
    """Load the model now, so the first real query does not wait for it.

    The model is loaded lazily, which would otherwise put several
    seconds onto whichever query happens to come first.  Calling this
    during startup moves that cost to where the user expects it.  It
    embeds one throwaway token and no corpus text, so nothing in the
    cache is embedded twice.
    """
    _encode([WARM_UP_TEXT])


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
