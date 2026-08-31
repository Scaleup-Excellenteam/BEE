"""Tests for production semantic dependency wiring."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main as application_main


def configure_startup_doubles(monkeypatch, index, semantic_store):
    initialize_corpus = Mock(return_value=index)
    set_corpus_index = Mock()
    open_or_build_store = Mock(return_value=semantic_store)
    run_cli = Mock()
    shutdown_logging = Mock()

    monkeypatch.setattr(application_main, "configure_logging", Mock())
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
    monkeypatch.setattr(
        application_main,
        "open_or_build_store",
        open_or_build_store,
    )
    monkeypatch.setattr(application_main, "run_cli", run_cli)
    monkeypatch.setattr(
        application_main,
        "shutdown_logging",
        shutdown_logging,
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

    return SimpleNamespace(
        initialize_corpus=initialize_corpus,
        set_corpus_index=set_corpus_index,
        open_or_build_store=open_or_build_store,
        run_cli=run_cli,
        shutdown_logging=shutdown_logging,
    )


def test_successful_semantic_startup_injects_real_interfaces(monkeypatch):
    index = SimpleNamespace(records=[object(), object()])
    semantic_store = Mock()
    doubles = configure_startup_doubles(
        monkeypatch,
        index,
        semantic_store,
    )

    application_main.main()

    doubles.initialize_corpus.assert_called_once_with("Archive.zip")
    doubles.set_corpus_index.assert_called_once_with(index)
    doubles.open_or_build_store.assert_called_once_with(index.records)
    doubles.run_cli.assert_called_once_with(
        semantic_search_fn=application_main.semantic_search_store,
        embedded_sentences=semantic_store,
        embedder=application_main.embed_text,
    )
    semantic_store.close.assert_called_once_with()
    doubles.shutdown_logging.assert_called_once_with()


def test_semantic_initialization_failure_keeps_part_a_available(monkeypatch):
    index = SimpleNamespace(records=[object()])
    doubles = configure_startup_doubles(monkeypatch, index, Mock())
    doubles.open_or_build_store.side_effect = RuntimeError("cache failed")

    application_main.main()

    doubles.set_corpus_index.assert_called_once_with(index)
    doubles.run_cli.assert_called_once_with()
    doubles.shutdown_logging.assert_called_once_with()


def test_semantic_store_closes_when_cli_fails(monkeypatch):
    index = SimpleNamespace(records=[object()])
    semantic_store = Mock()
    doubles = configure_startup_doubles(
        monkeypatch,
        index,
        semantic_store,
    )
    doubles.run_cli.side_effect = RuntimeError("CLI failed")

    with pytest.raises(RuntimeError, match="CLI failed"):
        application_main.main()

    semantic_store.close.assert_called_once_with()
    doubles.shutdown_logging.assert_called_once_with()
