"""Tests for the Spec 3 command-line interface."""

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
    fake_input = Mock(
        side_effect=[
            "this",
            " is",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    with pytest.raises(EOFError):
        cli.main()

    output = capsys.readouterr().out

    assert "The system is ready. Enter your text:" in output
    assert get_best_k_completions.call_args_list == [
        call("this"),
        call("this is"),
    ]


def test_hash_resets_input_without_running_autocomplete_for_hash(
    monkeypatch,
):
    fake_input = Mock(
        side_effect=[
            "hello",
            "#",
            "world",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    with pytest.raises(EOFError):
        cli.main()

    assert get_best_k_completions.call_args_list == [
        call("hello"),
        call("world"),
    ]


def test_displays_all_result_fields(monkeypatch, capsys):
    fake_input = Mock(
        side_effect=[
            "prefix",
            EOFError(),
        ]
    )
    result = FakeAutoCompleteData(
        completed_sentence="Completed sentence text",
        source_text="corpus/source.txt",
        offset=42,
        score=-3,
    )
    get_best_k_completions = Mock(return_value=[result])

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    with pytest.raises(EOFError):
        cli.main()

    output = capsys.readouterr().out

    assert "Completed sentence: Completed sentence text" in output
    assert "Source: corpus/source.txt" in output
    assert "Offset: 42" in output
    assert "Score: -3" in output


def test_cli_does_not_call_lower_level_dependencies(monkeypatch):
    fake_input = Mock(
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

    monkeypatch.setattr("builtins.input", fake_input)
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

    with pytest.raises(EOFError):
        cli.main()

    get_best_k_completions.assert_called_once_with("prefix")
    normalize_text.assert_not_called()
    corpus_index.assert_not_called()
    get_candidates.assert_not_called()
    calculate_best_match.assert_not_called()
