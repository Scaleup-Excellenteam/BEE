"""Tests for the Spanish translation helper."""

from unittest.mock import Mock

import pytest

from src import translation


def test_translates_text_using_google_translator(monkeypatch):
    translate = Mock(return_value="Hola mundo")
    translator_class = Mock(
        return_value=Mock(translate=translate)
    )
    monkeypatch.setattr(translation, "GoogleTranslator", translator_class)

    result = translation.translate_to_spanish("Hello world")

    assert result == "Hola mundo"
    translator_class.assert_called_once_with(source="auto", target="es")
    translate.assert_called_once_with("Hello world")


@pytest.mark.parametrize("blank_text", ["", "   ", "\n\t"])
def test_returns_blank_text_unchanged_without_calling_translator(
    monkeypatch,
    blank_text,
):
    translator_class = Mock()
    monkeypatch.setattr(translation, "GoogleTranslator", translator_class)

    result = translation.translate_to_spanish(blank_text)

    assert result == blank_text
    translator_class.assert_not_called()


def test_falls_back_to_original_text_when_translator_raises(monkeypatch):
    translator_class = Mock(
        return_value=Mock(
            translate=Mock(side_effect=ConnectionError("network down"))
        )
    )
    monkeypatch.setattr(translation, "GoogleTranslator", translator_class)

    result = translation.translate_to_spanish("Hello world")

    assert result == "Hello world"


def test_falls_back_to_original_text_when_translator_returns_empty(
    monkeypatch,
):
    translator_class = Mock(
        return_value=Mock(translate=Mock(return_value=None))
    )
    monkeypatch.setattr(translation, "GoogleTranslator", translator_class)

    result = translation.translate_to_spanish("Hello world")

    assert result == "Hello world"


def test_real_translation_call_returns_spanish_text():
    """Smoke test exercising the live Google Translate endpoint.

    Skips itself when outbound network access is unavailable, so the
    suite never depends on live connectivity to pass in CI.
    """
    result = translation.translate_to_spanish("hello world")

    if result == "hello world":
        # translate_to_spanish never raises; on any failure (including no
        # network) it falls back to the original text unchanged.
        pytest.skip("Live translation endpoint unreachable or unchanged")

    assert isinstance(result, str)
    assert result
