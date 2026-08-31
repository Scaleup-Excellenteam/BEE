"""Tests for Spec 3 autocomplete orchestration."""

from dataclasses import dataclass
from unittest.mock import Mock, call

import pytest

import autocomplete


@dataclass
class FakeSentenceRecord:
    original_sentence: str
    normalized_sentence: str
    source_text: str
    offset: int


@dataclass
class FakeAutoCompleteData:
    completed_sentence: str
    source_text: str
    offset: int
    score: int


@pytest.fixture
def fake_dependencies(monkeypatch):
    normalize_text = Mock()
    calculate_best_match = Mock()
    auto_complete_data = Mock(side_effect=FakeAutoCompleteData)

    monkeypatch.setattr(
        autocomplete,
        "normalize_text",
        normalize_text,
        raising=False,
    )
    monkeypatch.setattr(
        autocomplete,
        "calculate_best_match",
        calculate_best_match,
        raising=False,
    )
    monkeypatch.setattr(
        autocomplete,
        "AutoCompleteData",
        auto_complete_data,
        raising=False,
    )

    return normalize_text, calculate_best_match, auto_complete_data


def test_raises_when_corpus_index_is_not_configured(monkeypatch):
    monkeypatch.setattr(autocomplete, "_corpus_index", None)

    with pytest.raises(
        RuntimeError,
        match="Corpus index has not been configured",
    ):
        autocomplete.get_best_k_completions("original prefix")


def test_returns_empty_without_retrieving_candidates_for_empty_query(
    fake_dependencies,
):
    normalize_text, calculate_best_match, auto_complete_data = (
        fake_dependencies
    )
    normalize_text.return_value = ""

    index = Mock()
    autocomplete.set_corpus_index(index)

    results = autocomplete.get_best_k_completions("original prefix")

    assert results == []
    normalize_text.assert_called_once_with("original prefix")
    index.get_candidates.assert_not_called()
    calculate_best_match.assert_not_called()
    auto_complete_data.assert_not_called()


def test_processes_every_candidate_and_keeps_zero_and_negative_scores(
    fake_dependencies,
):
    normalize_text, calculate_best_match, auto_complete_data = (
        fake_dependencies
    )
    normalize_text.return_value = "normalized query"

    candidates = [
        FakeSentenceRecord("Skipped", "skipped", "a.txt", 10),
        FakeSentenceRecord("Zero", "zero", "b.txt", 20),
        FakeSentenceRecord("Negative", "negative", "c.txt", 30),
        FakeSentenceRecord("Positive", "positive", "d.txt", 40),
    ]

    index = Mock()
    index.get_candidates.return_value = candidates
    autocomplete.set_corpus_index(index)

    calculate_best_match.side_effect = [None, 0, -3, 8]

    results = autocomplete.get_best_k_completions("Original Prefix!")

    normalize_text.assert_called_once_with("Original Prefix!")
    index.get_candidates.assert_called_once_with("normalized query")
    assert calculate_best_match.call_args_list == [
        call("normalized query", "skipped"),
        call("normalized query", "zero"),
        call("normalized query", "negative"),
        call("normalized query", "positive"),
    ]

    assert auto_complete_data.call_args_list == [
        call(
            completed_sentence="Zero",
            source_text="b.txt",
            offset=20,
            score=0,
        ),
        call(
            completed_sentence="Negative",
            source_text="c.txt",
            offset=30,
            score=-3,
        ),
        call(
            completed_sentence="Positive",
            source_text="d.txt",
            offset=40,
            score=8,
        ),
    ]

    assert results == [
        FakeAutoCompleteData("Positive", "d.txt", 40, 8),
        FakeAutoCompleteData("Zero", "b.txt", 20, 0),
        FakeAutoCompleteData("Negative", "c.txt", 30, -3),
    ]


def test_sorts_by_score_then_sentence_and_returns_at_most_five(
    fake_dependencies,
):
    normalize_text, calculate_best_match, _ = fake_dependencies
    normalize_text.return_value = "query"

    candidates = [
        FakeSentenceRecord("Zulu", "zulu", "z.txt", 1),
        FakeSentenceRecord("Beta", "beta", "b.txt", 2),
        FakeSentenceRecord("Alpha", "alpha", "a.txt", 3),
        FakeSentenceRecord("Delta", "delta", "d.txt", 4),
        FakeSentenceRecord("Echo", "echo", "e.txt", 5),
        FakeSentenceRecord("Foxtrot", "foxtrot", "f.txt", 6),
        FakeSentenceRecord("Golf", "golf", "g.txt", 7),
    ]

    index = Mock()
    index.get_candidates.return_value = candidates
    autocomplete.set_corpus_index(index)

    scores = {
        "zulu": 10,
        "beta": 10,
        "alpha": 10,
        "delta": 9,
        "echo": 8,
        "foxtrot": 7,
        "golf": 6,
    }
    calculate_best_match.side_effect = (
        lambda query, sentence: scores[sentence]
    )

    results = autocomplete.get_best_k_completions("prefix")

    assert len(results) == 5
    assert [result.completed_sentence for result in results] == [
        "Alpha",
        "Beta",
        "Zulu",
        "Delta",
        "Echo",
    ]


def test_returns_fewer_than_five_when_fewer_valid_matches_exist(
    fake_dependencies,
):
    normalize_text, calculate_best_match, _ = fake_dependencies
    normalize_text.return_value = "query"

    candidates = [
        FakeSentenceRecord("One", "one", "one.txt", 1),
        FakeSentenceRecord("Two", "two", "two.txt", 2),
        FakeSentenceRecord("Three", "three", "three.txt", 3),
    ]

    index = Mock()
    index.get_candidates.return_value = candidates
    autocomplete.set_corpus_index(index)

    calculate_best_match.side_effect = [4, None, 2]

    results = autocomplete.get_best_k_completions("prefix")

    assert len(results) == 2
    assert [result.completed_sentence for result in results] == [
        "One",
        "Three",
    ]


def test_reuses_the_configured_index_reference(fake_dependencies):
    normalize_text, calculate_best_match, _ = fake_dependencies
    normalize_text.side_effect = ["first query", "second query"]
    calculate_best_match.return_value = 1

    candidate = FakeSentenceRecord(
        "Result",
        "result",
        "source.txt",
        12,
    )
    index = Mock()
    index.get_candidates.return_value = [candidate]

    autocomplete.set_corpus_index(index)

    autocomplete.get_best_k_completions("first")
    autocomplete.get_best_k_completions("second")

    assert index.get_candidates.call_args_list == [
        call("first query"),
        call("second query"),
    ]
