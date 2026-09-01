"""Tests for optional Translation mode in BEE's existing CLI."""

from dataclasses import dataclass
from unittest.mock import Mock, call

import cli
from src.translation import TranslationError, TranslationResult


HEBREW_HELLO = "\u05e9\u05dc\u05d5\u05dd"
HEBREW_PYTHON_FUNCTION = (
    "\u05e4\u05d5\u05e0\u05e7\u05e6\u05d9\u05d9\u05ea "
    "\u05e4\u05d9\u05d9\u05ea\u05d5\u05df"
)


@dataclass
class FakeAutoCompleteData:
    completed_sentence: str
    source_text: str
    offset: int
    score: int


class FakeTranslator:
    def __init__(
        self,
        translated_text="python function",
        error=None,
        localized_texts=None,
        localization_error=None,
    ):
        self.translated_text = translated_text
        self.error = error
        self.calls = []
        self.localized_texts = localized_texts or {}
        self.localization_error = localization_error
        self.localization_calls = []

    def translate_to_english(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return TranslationResult(
            original_text=text,
            translated_text=self.translated_text,
            detected_source_language="he",
        )

    def translate(self, text, target_language):
        self.localization_calls.append((text, target_language))
        if self.localization_error is not None:
            raise self.localization_error
        return TranslationResult(
            original_text=text,
            translated_text=self.localized_texts.get(text, text),
            detected_source_language="en",
        )


def _run_cli(monkeypatch, inputs, search_results, translation_service=None):
    read_prefilled_input = Mock(side_effect=[*inputs, EOFError()])
    get_best_k_completions = Mock(return_value=search_results)
    monkeypatch.setattr(cli, "_read_prefilled_input", read_prefilled_input)
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.main(translation_service=translation_service)

    return read_prefilled_input, get_best_k_completions


def test_translate_command_enables_mode_without_search(monkeypatch, capsys):
    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND],
        [],
    )

    output = capsys.readouterr().out
    assert cli.TRANSLATION_MODE_ENABLED in output
    get_best_k_completions.assert_not_called()


def test_spanish_command_enables_localization_without_search(
    monkeypatch,
    capsys,
):
    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND],
        [],
    )

    assert cli.SPANISH_MODE_ENABLED in capsys.readouterr().out
    get_best_k_completions.assert_not_called()


def test_translation_mode_translates_then_searches_and_displays_query(
    monkeypatch,
    capsys,
):
    translator = FakeTranslator("python function")
    result = FakeAutoCompleteData(
        completed_sentence="Python function docs",
        source_text="python.txt",
        offset=3,
        score=30,
    )

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, HEBREW_PYTHON_FUNCTION],
        [result],
        translator,
    )

    output = capsys.readouterr().out
    assert translator.calls == [HEBREW_PYTHON_FUNCTION]
    get_best_k_completions.assert_called_once_with("python function")
    assert "Translated English query: python function" in output
    assert "1. Python function docs (python.txt:3, score=30)" in output


def test_translation_failure_does_not_search_or_crash(monkeypatch, capsys):
    translator = FakeTranslator(
        error=TranslationError("request timed out")
    )

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, HEBREW_HELLO],
        [],
        translator,
    )

    assert "Translation failed: request timed out" in capsys.readouterr().out
    get_best_k_completions.assert_not_called()


def test_translation_mode_without_service_is_clear(monkeypatch, capsys):
    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, HEBREW_HELLO],
        [],
    )

    assert cli.TRANSLATION_UNAVAILABLE in capsys.readouterr().out
    get_best_k_completions.assert_not_called()


def test_translation_mode_reports_no_autocomplete_results(
    monkeypatch,
    capsys,
):
    translator = FakeTranslator("missing phrase")

    _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, HEBREW_HELLO],
        [],
        translator,
    )

    output = capsys.readouterr().out
    assert "Translated English query: missing phrase" in output
    assert "No suggestions found." in output


def test_normal_part_a_mode_remains_unchanged(monkeypatch):
    translator = FakeTranslator()

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        ["original English query"],
        [],
        translator,
    )

    assert translator.calls == []
    assert translator.localization_calls == []
    get_best_k_completions.assert_called_once_with(
        "original English query"
    )


def test_spanish_mode_localizes_results_without_mutating_metadata(
    monkeypatch,
    capsys,
):
    results = [
        FakeAutoCompleteData(
            completed_sentence="To be or not to be",
            source_text="hamlet.txt",
            offset=42,
            score=14,
        ),
        FakeAutoCompleteData(
            completed_sentence="All the world's a stage",
            source_text="as-you-like-it.txt",
            offset=7,
            score=10,
        ),
    ]
    translator = FakeTranslator(
        localized_texts={
            "To be or not to be": "Ser o no ser",
            "All the world's a stage": "Todo el mundo es un escenario",
        }
    )

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND, "English query"],
        results,
        translator,
    )

    output = capsys.readouterr().out
    get_best_k_completions.assert_called_once_with("English query")
    assert translator.calls == []
    assert translator.localization_calls == [
        ("To be or not to be", "es"),
        ("All the world's a stage", "es"),
    ]
    assert "1. Original: To be or not to be" in output
    assert "   Spanish: Ser o no ser" in output
    assert "   Source: hamlet.txt" in output
    assert "   Offset: 42" in output
    assert "   Score: 14" in output
    assert "2. Original: All the world's a stage" in output
    assert "   Spanish: Todo el mundo es un escenario" in output
    assert output.index("1. Original:") < output.index("2. Original:")
    assert results == [
        FakeAutoCompleteData(
            completed_sentence="To be or not to be",
            source_text="hamlet.txt",
            offset=42,
            score=14,
        ),
        FakeAutoCompleteData(
            completed_sentence="All the world's a stage",
            source_text="as-you-like-it.txt",
            offset=7,
            score=10,
        ),
    ]


def test_spanish_localization_failure_falls_back_to_original_result(
    monkeypatch,
    capsys,
):
    result = FakeAutoCompleteData(
        completed_sentence="Original English sentence",
        source_text="source.txt",
        offset=9,
        score=-2,
    )
    translator = FakeTranslator(
        localization_error=TranslationError("request timed out")
    )

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND, "query"],
        [result],
        translator,
    )

    output = capsys.readouterr().out
    get_best_k_completions.assert_called_once_with("query")
    assert "1. Original: Original English sentence" in output
    assert "   Spanish: unavailable" in output
    assert "   Source: source.txt" in output
    assert "   Offset: 9" in output
    assert "   Score: -2" in output


def test_spanish_mode_without_service_reports_unavailable_once_per_search(
    monkeypatch,
    capsys,
):
    results = [
        FakeAutoCompleteData("First", "first.txt", 1, 5),
        FakeAutoCompleteData("Second", "second.txt", 2, 4),
    ]

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND, "query"],
        results,
    )

    output = capsys.readouterr().out
    get_best_k_completions.assert_called_once_with("query")
    assert output.count(cli.SPANISH_LOCALIZATION_UNAVAILABLE) == 1
    assert "1. Original: First" in output
    assert "2. Original: Second" in output
    assert output.count("Spanish: unavailable") == 2


def test_english_command_returns_to_normal_part_a_mode(monkeypatch, capsys):
    translator = FakeTranslator()

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, cli.ENGLISH_COMMAND, "English query"],
        [],
        translator,
    )

    assert cli.ENGLISH_MODE_ENABLED in capsys.readouterr().out
    assert translator.calls == []
    get_best_k_completions.assert_called_once_with("English query")


def test_english_command_disables_spanish_localization(monkeypatch, capsys):
    translator = FakeTranslator(
        localized_texts={"Original": "Localizado"}
    )
    result = FakeAutoCompleteData("Original", "source.txt", 1, 4)

    _, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND, cli.ENGLISH_COMMAND, "English query"],
        [result],
        translator,
    )

    output = capsys.readouterr().out
    assert translator.localization_calls == []
    get_best_k_completions.assert_called_once_with("English query")
    assert "1. Original (source.txt:1, score=4)" in output
    assert "Spanish:" not in output


def test_hash_reset_stays_in_translation_mode_and_clears_query(
    monkeypatch,
):
    translator = FakeTranslator("hello")

    read_prefilled_input, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.TRANSLATE_COMMAND, "old query#", HEBREW_HELLO],
        [],
        translator,
    )

    assert read_prefilled_input.call_args_list == [
        call(""),
        call(""),
        call(""),
        call(HEBREW_HELLO),
    ]
    assert translator.calls == [HEBREW_HELLO]
    get_best_k_completions.assert_called_once_with("hello")


def test_hash_reset_stays_in_spanish_mode_and_clears_query(monkeypatch):
    translator = FakeTranslator(
        localized_texts={"Result": "Resultado"}
    )
    result = FakeAutoCompleteData("Result", "source.txt", 1, 5)

    read_prefilled_input, get_best_k_completions = _run_cli(
        monkeypatch,
        [cli.SPANISH_COMMAND, "old query#", "new query"],
        [result],
        translator,
    )

    assert read_prefilled_input.call_args_list == [
        call(""),
        call(""),
        call(""),
        call("new query"),
    ]
    get_best_k_completions.assert_called_once_with("new query")
    assert translator.localization_calls == [("Result", "es")]
