"""Tests that Translation presentation remains outside the Part A cache."""

from unittest.mock import Mock

import autocomplete
import cli
from src.models import SentenceRecord
from src.translation import TranslationResult


class FakeTranslationService:
    def __init__(self, english_translations=None):
        self.english_translations = iter(english_translations or [])
        self.english_calls = []
        self.localization_calls = []

    def translate_to_english(self, text):
        self.english_calls.append(text)
        return TranslationResult(
            original_text=text,
            translated_text=next(self.english_translations),
            detected_source_language="he",
        )

    def translate(self, text, target_language):
        self.localization_calls.append((text, target_language))
        return TranslationResult(
            original_text=text,
            translated_text="Hola mundo",
            detected_source_language="en",
        )


def _configure_real_autocomplete(monkeypatch):
    index = Mock()
    index.get_candidates.return_value = [
        SentenceRecord(
            original_sentence="Hello world",
            normalized_sentence="hello world",
            source_text="greetings.txt",
            offset=7,
        )
    ]
    autocomplete.set_corpus_index(index)
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        autocomplete.get_best_k_completions,
    )
    return index


def test_translate_repeats_service_call_but_reuses_part_a_cache(
    monkeypatch,
):
    index = _configure_real_autocomplete(monkeypatch)
    service = FakeTranslationService(["HELLO!", " hello "])
    read_input = Mock(side_effect=["/translate", "שלום", "עולם", EOFError])
    monkeypatch.setattr(cli, "_read_prefilled_input", read_input)

    cli.main(service)

    assert service.english_calls == ["שלום", "עולם"]
    index.get_candidates.assert_called_once_with("hello")
    info = autocomplete.get_query_cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_spanish_reuses_part_a_cache_but_localizes_each_display(
    monkeypatch,
):
    index = _configure_real_autocomplete(monkeypatch)
    service = FakeTranslationService()
    read_input = Mock(side_effect=["/spanish", "HELLO!", " hello ", EOFError])
    monkeypatch.setattr(cli, "_read_prefilled_input", read_input)

    cli.main(service)

    index.get_candidates.assert_called_once_with("hello")
    assert service.localization_calls == [
        ("Hello world", "es"),
        ("Hello world", "es"),
    ]
    info = autocomplete.get_query_cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_switching_spanish_to_english_never_caches_localized_text(
    monkeypatch,
    capsys,
):
    index = _configure_real_autocomplete(monkeypatch)
    service = FakeTranslationService()
    read_input = Mock(
        side_effect=["/spanish", "hello", "/english", "HELLO!", EOFError]
    )
    monkeypatch.setattr(cli, "_read_prefilled_input", read_input)

    cli.main(service)

    output = capsys.readouterr().out
    assert "Spanish: Hola mundo" in output
    assert "2. Hola mundo" not in output
    assert "1. Hello world (greetings.txt:7, score=10)" in output
    assert service.localization_calls == [("Hello world", "es")]
    index.get_candidates.assert_called_once_with("hello")

    cached_results = autocomplete.get_best_k_completions("hello")
    assert cached_results[0].completed_sentence == "Hello world"
