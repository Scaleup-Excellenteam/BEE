"""Tests for optional Translation initialization during BEE startup."""

import sys
from unittest.mock import Mock

import main as application_main


def _prepare_main(monkeypatch, translation_service):
    index = object()
    initialize_corpus = Mock(return_value=index)
    set_corpus_index = Mock()
    run_cli = Mock()

    monkeypatch.setattr(application_main, "configure_logging", Mock())
    monkeypatch.setattr(application_main, "shutdown_logging", Mock())
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
    # Semantic Fault Memory is exercised in tests/test_main_log_search.py.
    # Stubbing it here keeps these translation tests from building a real
    # service, loading the local model, or writing to logs/.
    monkeypatch.setattr(
        application_main,
        "initialize_log_search",
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        application_main,
        "_initialize_translation_service",
        Mock(return_value=translation_service),
    )
    monkeypatch.setattr(
        application_main.time,
        "perf_counter",
        Mock(side_effect=[10.0, 12.5]),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

    return index, initialize_corpus, set_corpus_index, run_cli


def test_translation_service_is_injected_after_part_a_startup(monkeypatch):
    translation_service = object()
    index, initialize_corpus, set_corpus_index, run_cli = _prepare_main(
        monkeypatch,
        translation_service,
    )

    application_main.main()

    initialize_corpus.assert_called_once_with("Archive.zip")
    set_corpus_index.assert_called_once_with(index)
    run_cli.assert_called_once_with(
        translation_service=translation_service
    )


def test_unavailable_translation_still_runs_part_a(monkeypatch):
    index, initialize_corpus, set_corpus_index, run_cli = _prepare_main(
        monkeypatch,
        None,
    )

    application_main.main()

    initialize_corpus.assert_called_once_with("Archive.zip")
    set_corpus_index.assert_called_once_with(index)
    run_cli.assert_called_once_with()


def test_translation_configuration_failure_returns_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        application_main,
        "GoogleTranslationService",
        Mock(side_effect=RuntimeError("sensitive configuration detail")),
    )

    assert application_main._initialize_translation_service() is None


def test_translation_dependency_check_does_not_create_google_client(
    monkeypatch,
):
    service = Mock(project_id="configured-project")
    service_factory = Mock(return_value=service)
    import_module = Mock(return_value=object())
    monkeypatch.setattr(
        application_main,
        "GoogleTranslationService",
        service_factory,
    )
    monkeypatch.setattr(application_main.importlib, "import_module", import_module)

    result = application_main._initialize_translation_service()

    assert result is service
    service_factory.assert_called_once_with()
    import_module.assert_called_once_with("google.cloud.translate_v3")
    assert service.method_calls == []
