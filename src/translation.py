"""Spanish translation of user-facing answers.

Wraps ``deep_translator``'s Google Translate backend behind a small,
failure-tolerant function so callers never need to worry about network
or translation-service errors: a failed translation simply falls back
to returning the original text unchanged.
"""

from __future__ import annotations

from deep_translator import GoogleTranslator

from src.logging_config import get_application_logger


LOGGER = get_application_logger()

TARGET_LANGUAGE = "es"


def translate_to_spanish(text: str) -> str:
    """Translate text to Spanish, falling back to the original on failure."""
    if not text or not text.strip():
        return text

    try:
        translated = GoogleTranslator(
            source="auto",
            target=TARGET_LANGUAGE,
        ).translate(text)
    except Exception as error:
        # Network failures, rate limiting, and any other translator error
        # must never break the caller. Fall back to the original text.
        LOGGER.warning(
            "Translation to Spanish failed for %r: %s",
            text,
            error,
        )
        return text

    if not translated:
        return text

    return translated
