"""Tests for wiring Semantic Search into application startup.

The semantic store is always mocked, so no test reaches Gemini.
"""

import sys
from unittest.mock import Mock

import pytest

import main as application_main
from src.logging_config import configure_logging, shutdown_logging
from src.models import SentenceRecord


@pytest.fixture(autouse=True)
def close_log_handlers_after_test():
    shutdown_logging()
    yield
    shutdown_logging()


def make_index(record_count):
    """Return an object shaped like a CorpusIndex with ``record_count``."""
    return Mock(
        records=[
            SentenceRecord(
                original_sentence=f"sentence number {index}",
                normalized_sentence=f"sentence number {index}",
                source_text="corpus.txt",
                offset=index,
            )
            for index in range(record_count)
        ]
    )


@pytest.fixture
def startup(monkeypatch, tmp_path):
    """Run main() with the corpus, CLI and log file all replaced."""
    log_path = tmp_path / "autocomplete.log"

    def run(index, open_or_build_store):
        run_cli = Mock()

        monkeypatch.setattr(
            application_main,
            "configure_logging",
            lambda: configure_logging(log_path),
        )
        monkeypatch.setattr(
            application_main,
            "initialize_corpus",
            Mock(return_value=index),
        )
        monkeypatch.setattr(application_main, "set_corpus_index", Mock())
        monkeypatch.setattr(
            application_main,
            "open_or_build_store",
            open_or_build_store,
        )
        monkeypatch.setattr(application_main, "run_cli", run_cli)
        monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

        application_main.main()
        shutdown_logging()

        return run_cli, log_path.read_text(encoding="utf-8")

    return run


def test_semantic_dependencies_are_passed_to_the_cli(startup):
    store = Mock(__len__=Mock(return_value=100))
    open_or_build_store = Mock(return_value=store)

    run_cli, _ = startup(make_index(150), open_or_build_store)

    run_cli.assert_called_once_with(
        semantic_search_fn=application_main.semantic_search,
        embedded_sentences=store,
        embedder=application_main.embed_text,
    )


def test_records_are_sampled_evenly_across_the_whole_corpus(startup):
    store = Mock(__len__=Mock(return_value=100))
    open_or_build_store = Mock(return_value=store)
    index = make_index(1000)

    startup(index, open_or_build_store)

    passed_records = open_or_build_store.call_args.args[0]
    assert len(passed_records) == 100
    # Every tenth sentence, reaching the end of the corpus rather than
    # stopping inside the first source file.
    assert [record.offset for record in passed_records] == list(
        range(0, 1000, 10)
    )


def test_the_sample_never_exceeds_the_sample_size(startup):
    store = Mock(__len__=Mock(return_value=100))
    open_or_build_store = Mock(return_value=store)

    # 1050 // 100 is a step of 10, which would otherwise yield 105 records.
    startup(make_index(1050), open_or_build_store)

    assert len(open_or_build_store.call_args.args[0]) == 100


def test_a_corpus_barely_larger_than_the_sample_takes_the_front(startup):
    store = Mock(__len__=Mock(return_value=100))
    open_or_build_store = Mock(return_value=store)

    # 150 // 100 floors to a step of 1, so this is simply the first 100.
    startup(make_index(150), open_or_build_store)

    passed_records = open_or_build_store.call_args.args[0]
    assert [record.offset for record in passed_records] == list(range(100))


def test_a_separate_test_cache_directory_is_used(startup):
    store = Mock(__len__=Mock(return_value=100))
    open_or_build_store = Mock(return_value=store)

    startup(make_index(150), open_or_build_store)

    assert (
        open_or_build_store.call_args.kwargs["cache_path"]
        == "embedding_cache_test"
    )
    assert (
        open_or_build_store.call_args.kwargs["batch_embedder"]
        is application_main.embed_texts
    )


def test_a_smaller_corpus_is_embedded_whole(startup):
    store = Mock(__len__=Mock(return_value=3))
    open_or_build_store = Mock(return_value=store)

    startup(make_index(3), open_or_build_store)

    assert len(open_or_build_store.call_args.args[0]) == 3


def test_the_store_is_closed_after_the_cli_exits(startup):
    store = Mock(__len__=Mock(return_value=100))

    startup(make_index(150), Mock(return_value=store))

    store.close.assert_called_once_with()


def test_semantic_failure_leaves_regular_autocomplete_working(startup):
    open_or_build_store = Mock(
        side_effect=RuntimeError("Missing Gemini API key")
    )

    run_cli, log_text = startup(make_index(150), open_or_build_store)

    run_cli.assert_called_once_with()
    assert (
        "Semantic Search is unavailable because its preparation failed: "
        "Missing Gemini API key"
        in log_text
    )
    assert "Traceback" in log_text
    assert "The autocomplete system is ready for searches." in log_text


def test_semantic_success_is_logged(startup):
    store = Mock(__len__=Mock(return_value=100))

    _, log_text = startup(make_index(150), Mock(return_value=store))

    assert (
        "Semantic preparation started using 100 corpus sentences." in log_text
    )
    assert (
        "Semantic preparation completed successfully with 100 sentences."
        in log_text
    )
