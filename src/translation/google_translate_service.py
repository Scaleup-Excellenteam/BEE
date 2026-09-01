"""Google Cloud Translation API wrapper.

Credentials are intentionally provided only through Google Application Default
Credentials and environment variables. No API keys or service-account details
belong in source code.
"""

import os
from typing import Optional

from .models import TranslationError, TranslationResult

DEFAULT_LOCATION = "global"
DEFAULT_TARGET_LANGUAGE = "en"
DEFAULT_TIMEOUT_SECONDS = 10.0


class GoogleTranslationService:
    """Translate text using Google Cloud Translation v3."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.timeout_seconds = timeout_seconds or self._timeout_from_environment()
        self._client = None

    def translate(
        self,
        text: str,
        target_language: str,
    ) -> TranslationResult:
        query = text.strip()
        if not query:
            raise TranslationError("translation input is empty")

        target = target_language.strip()
        if not target:
            raise TranslationError("target language is empty")

        if not self.project_id:
            raise TranslationError("GOOGLE_CLOUD_PROJECT is not configured")

        try:
            client = self._get_client()
            response = client.translate_text(
                request={
                    "parent": f"projects/{self.project_id}/locations/{self.location}",
                    "contents": [query],
                    "mime_type": "text/plain",
                    "target_language_code": target,
                },
                timeout=self.timeout_seconds,
            )
        except ImportError as exc:
            raise TranslationError(
                "google-cloud-translate is not installed"
            ) from exc
        except Exception as exc:
            raise TranslationError(f"Google Translation API request failed: {exc}") from exc

        translations = getattr(response, "translations", None)
        if not translations:
            raise TranslationError("Google Translation API returned no translations")

        first_translation = translations[0]
        translated_text = getattr(first_translation, "translated_text", "")
        if not translated_text or not translated_text.strip():
            raise TranslationError("Google Translation API returned an empty translation")

        detected_language = getattr(first_translation, "detected_language_code", None)
        return TranslationResult(
            original_text=query,
            translated_text=translated_text.strip(),
            detected_source_language=detected_language,
        )

    def translate_to_english(self, text: str) -> TranslationResult:
        """Translate text to English using source-language auto-detection."""
        return self.translate(text, DEFAULT_TARGET_LANGUAGE)

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import translate_v3
            except ImportError as exc:
                raise ImportError("google-cloud-translate is required") from exc
            self._client = translate_v3.TranslationServiceClient()
        return self._client

    @staticmethod
    def _timeout_from_environment() -> float:
        raw_timeout = os.environ.get("TRANSLATION_TIMEOUT_SECONDS")
        if raw_timeout is None:
            return DEFAULT_TIMEOUT_SECONDS
        try:
            return float(raw_timeout)
        except ValueError:
            return DEFAULT_TIMEOUT_SECONDS
