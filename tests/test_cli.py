"""Tests for the Spec 3 command-line interface."""

import sys
from dataclasses import dataclass
from unittest.mock import Mock, call

import pytest

import cli


@dataclass
class FakeAutoCompleteData:
    completed_sentence: str
    source_text: str
    offset: int
    score: int


def test_prints_readiness_message_and_accumulates_input_exactly(
    monkeypatch,
    capsys,
):
    read_prefilled_input = Mock(
        side_effect=[
            "this",
            "this is",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.main()

    output = capsys.readouterr().out

    assert output.splitlines() == [
        "The system is ready. Enter your text:",
        "No suggestions found.",
        "No suggestions found.",
    ]
    assert "Current text:" not in output
    assert "Continue typing:" not in output
    assert read_prefilled_input.call_args_list == [
        call(""),
        call("this"),
        call("this is"),
    ]
    assert get_best_k_completions.call_args_list == [
        call("this"),
        call("this is"),
    ]


@pytest.mark.parametrize(
    "terminated_input",
    [
        "#",
        "this is the fix option#",
        "this is #",
        "hello#world",
    ],
)
def test_hash_anywhere_resets_without_running_autocomplete(
    monkeypatch,
    terminated_input,
):
    read_prefilled_input = Mock(
        side_effect=[
            terminated_input,
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.main()

    get_best_k_completions.assert_not_called()
    assert read_prefilled_input.call_args_list == [
        call(""),
        call(""),
    ]


def test_text_after_hash_starts_from_empty_state(monkeypatch):
    read_prefilled_input = Mock(
        side_effect=[
            "this is#",
            "hello",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.main()

    get_best_k_completions.assert_called_once_with("hello")
    assert read_prefilled_input.call_args_list == [
        call(""),
        call(""),
        call("hello"),
    ]


def _run_cli_with_results(
    monkeypatch,
    capsys,
    results,
    translate_to_spanish=lambda text: text,
):
    """Run one query through the CLI and return everything it printed."""
    read_prefilled_input = Mock(
        side_effect=[
            "prefix",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=results)

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )
    monkeypatch.setattr(
        cli,
        "translate_to_spanish",
        translate_to_spanish,
    )

    cli.main()

    return capsys.readouterr().out


def test_displays_all_result_fields(monkeypatch, capsys):
    result = FakeAutoCompleteData(
        completed_sentence="Completed sentence text",
        source_text="corpus/source.txt",
        offset=42,
        score=-3,
    )

    output = _run_cli_with_results(monkeypatch, capsys, [result])

    assert "Here are 1 suggestions:" in output
    assert (
        "1. Completed sentence text (corpus/source.txt:42, score=-3)"
        in output
    )


def test_numbers_suggestions_in_returned_order(monkeypatch, capsys):
    results = [
        FakeAutoCompleteData(
            completed_sentence=f"Sentence {position}",
            source_text="example.txt",
            offset=position,
            score=14,
        )
        for position in range(1, 6)
    ]

    output = _run_cli_with_results(monkeypatch, capsys, results)

    assert "Here are 5 suggestions:" in output
    for position in range(1, 6):
        assert (
            f"{position}. Sentence {position} "
            f"(example.txt:{position}, score=14)"
        ) in output


def test_translates_each_completed_sentence_before_printing(
    monkeypatch,
    capsys,
):
    results = [
        FakeAutoCompleteData(
            completed_sentence=f"Sentence {position}",
            source_text="example.txt",
            offset=position,
            score=14,
        )
        for position in range(1, 3)
    ]
    translate_to_spanish = Mock(
        side_effect=lambda text: f"[ES] {text}"
    )

    output = _run_cli_with_results(
        monkeypatch,
        capsys,
        results,
        translate_to_spanish=translate_to_spanish,
    )

    assert translate_to_spanish.call_args_list == [
        call("Sentence 1"),
        call("Sentence 2"),
    ]
    assert "1. [ES] Sentence 1 (example.txt:1, score=14)" in output
    assert "2. [ES] Sentence 2 (example.txt:2, score=14)" in output


def test_falls_back_to_original_sentence_when_translation_fails(
    monkeypatch,
    capsys,
):
    result = FakeAutoCompleteData(
        completed_sentence="Untranslatable sentence",
        source_text="example.txt",
        offset=1,
        score=10,
    )

    output = _run_cli_with_results(
        monkeypatch,
        capsys,
        [result],
        translate_to_spanish=lambda text: text,
    )

    assert (
        "1. Untranslatable sentence (example.txt:1, score=10)" in output
    )


def test_reports_when_no_suggestions_are_found(monkeypatch, capsys):
    output = _run_cli_with_results(monkeypatch, capsys, [])

    assert "No suggestions found." in output
    assert "suggestions:" not in output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console test")
def test_windows_input_prefills_the_editable_line(monkeypatch, capsys):
    import msvcrt

    characters = iter([" ", "i", "s", "\r"])
    monkeypatch.setattr(msvcrt, "getwch", lambda: next(characters))

    result = cli._read_windows_prefilled_input("this")

    assert result == "this is"
    assert capsys.readouterr().out == "this is\n"


@pytest.mark.parametrize("termination", [EOFError(), KeyboardInterrupt()])
def test_normal_cli_termination_is_clean(monkeypatch, termination):
    read_prefilled_input = Mock(side_effect=termination)
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.main()

    get_best_k_completions.assert_not_called()


def test_cli_does_not_call_lower_level_dependencies(monkeypatch):
    read_prefilled_input = Mock(
        side_effect=[
            "prefix",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])
    normalize_text = Mock()
    corpus_index = Mock()
    get_candidates = Mock()
    calculate_best_match = Mock()

    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )
    monkeypatch.setattr(
        cli,
        "normalize_text",
        normalize_text,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "CorpusIndex",
        corpus_index,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "get_candidates",
        get_candidates,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "calculate_best_match",
        calculate_best_match,
        raising=False,
    )

    cli.main()

    get_best_k_completions.assert_called_once_with("prefix")
    normalize_text.assert_not_called()
    corpus_index.assert_not_called()
    get_candidates.assert_not_called()
    calculate_best_match.assert_not_called()
