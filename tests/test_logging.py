"""Tests for human-readable runtime logging."""

import re
import sys
from dataclasses import dataclass
from unittest.mock import Mock, call

import pytest

import autocomplete
import cli
import main as application_main
from src.corpus.index import CorpusIndex
from src.logging_config import configure_logging, shutdown_logging
from src.models import AutoCompleteData, SentenceRecord


@dataclass
class FakeAutoCompleteData:
    completed_sentence: str
    source_text: str
    offset: int
    score: int


@pytest.fixture(autouse=True)
def close_log_handlers_after_test():
    shutdown_logging()
    yield
    shutdown_logging()


def test_configure_logging_creates_timestamped_log_file(tmp_path):
    log_path = tmp_path / "logs" / "autocomplete.log"

    logger = configure_logging(log_path)
    logger.info("Application started.")
    shutdown_logging()

    log_text = log_path.read_text(encoding="utf-8")
    assert re.search(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| ",
        log_text,
    )
    assert "Application started." in log_text


def test_main_logs_application_and_corpus_lifecycle(
    monkeypatch,
    tmp_path,
):
    log_path = tmp_path / "autocomplete.log"
    index = object()
    initialize_corpus = Mock(return_value=index)
    set_corpus_index = Mock()
    run_cli = Mock()

    monkeypatch.setattr(
        application_main,
        "configure_logging",
        lambda: configure_logging(log_path),
    )
    monkeypatch.setattr(
        application_main,
        "initialize_corpus",
        initialize_corpus,
    )
    monkeypatch.setattr(
        application_main,
        "set_corpus_index",
        set_corpus_index,
    )
    monkeypatch.setattr(application_main, "run_cli", run_cli)
    monkeypatch.setattr(
        application_main.time,
        "perf_counter",
        Mock(side_effect=[10.0, 12.5]),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

    application_main.main()

    initialize_corpus.assert_called_once_with("Archive.zip")
    set_corpus_index.assert_called_once_with(index)
    run_cli.assert_called_once_with()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Application started." in log_text
    assert 'Corpus preparation started using "Archive.zip".' in log_text
    assert (
        "Corpus preparation completed successfully in 2.500 seconds."
        in log_text
    )
    assert "The autocomplete system is ready for searches." in log_text


def test_corpus_initialization_error_is_logged(monkeypatch, tmp_path):
    log_path = tmp_path / "autocomplete.log"

    monkeypatch.setattr(
        application_main,
        "configure_logging",
        lambda: configure_logging(log_path),
    )
    monkeypatch.setattr(
        application_main,
        "initialize_corpus",
        Mock(side_effect=RuntimeError("archive could not be opened")),
    )
    monkeypatch.setattr(
        application_main.time,
        "perf_counter",
        Mock(side_effect=[4.0, 5.0]),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "broken.zip"])

    with pytest.raises(RuntimeError, match="archive could not be opened"):
        application_main.main()

    log_text = log_path.read_text(encoding="utf-8")
    assert (
        "An error occurred while preparing the corpus after 1.000 seconds: "
        "archive could not be opened"
        in log_text
    )


def test_cli_logs_query_reset_and_search_results_without_changing_output(
    monkeypatch,
    capsys,
    tmp_path,
):
    log_path = tmp_path / "autocomplete.log"
    configure_logging(log_path)

    result = FakeAutoCompleteData(
        completed_sentence="To be or not to be",
        source_text="hamlet.txt",
        offset=42,
        score=10,
    )
    read_prefilled_input = Mock(
        side_effect=["to be", "unknown", "#", EOFError()]
    )
    get_best_k_completions = Mock(side_effect=[[result], []])

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
        cli.time,
        "perf_counter",
        Mock(side_effect=[1.0, 1.25, 2.0, 2.5]),
    )

    cli.main()
    shutdown_logging()

    assert read_prefilled_input.call_args_list == [
        call(""),
        call("to be"),
        call("unknown"),
        call(""),
    ]
    assert get_best_k_completions.call_args_list == [
        call("to be"),
        call("unknown"),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "The system is ready. Enter your text:",
        "Here are 1 suggestions:",
        "1. To be or not to be (hamlet.txt:42, score=10)",
        "No suggestions found.",
    ]

    log_text = log_path.read_text(encoding="utf-8")
    assert 'User submitted a search query: "to be"' in log_text
    assert (
        "Search completed successfully in 0.250 seconds. "
        "1 suggestion was returned."
        in log_text
    )
    assert 'User submitted a search query: "unknown"' in log_text
    assert (
        "Search completed successfully in 0.500 seconds. "
        "No suggestions were found for the current query."
        in log_text
    )
    assert (
        "The user finished the current query and started a new one."
        in log_text
    )
    assert "Application closed by the user." in log_text


def test_logging_does_not_change_autocomplete_results(monkeypatch, tmp_path):
    log_path = tmp_path / "autocomplete.log"
    configure_logging(log_path)

    records = [
        SentenceRecord(
            original_sentence="Hello World",
            normalized_sentence="hello world",
            source_text="greetings.txt",
            offset=1,
        )
    ]
    monkeypatch.setattr(autocomplete, "_corpus_index", CorpusIndex(records))

    results = autocomplete.get_best_k_completions("HELLO!")

    assert results == [
        AutoCompleteData(
            completed_sentence="Hello World",
            source_text="greetings.txt",
            offset=1,
            score=10,
        )
    ]
