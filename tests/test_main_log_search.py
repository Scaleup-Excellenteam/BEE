"""Tests for wiring Semantic Log Search into application startup.

``LogSearchService`` is always replaced, so no test loads the local
model, touches the real cache, or writes to the real log.
"""

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

import main as application_main
from src.logging_config import configure_logging, shutdown_logging


@pytest.fixture(autouse=True)
def close_log_handlers_after_test():
    shutdown_logging()
    yield
    shutdown_logging()


@pytest.fixture
def startup(monkeypatch, tmp_path):
    """Run main() with the corpus, CLI and log file all replaced."""
    log_path = tmp_path / "autocomplete.log"

    def run(service_factory):
        run_cli = Mock()

        monkeypatch.setattr(
            application_main,
            "configure_logging",
            lambda: configure_logging(log_path),
        )
        monkeypatch.setattr(
            application_main,
            "initialize_corpus",
            Mock(return_value=object()),
        )
        monkeypatch.setattr(application_main, "set_corpus_index", Mock())
        monkeypatch.setattr(
            application_main,
            "LogSearchService",
            service_factory,
        )
        monkeypatch.setattr(application_main, "run_cli", run_cli)
        monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

        application_main.main()
        shutdown_logging()

        return run_cli, log_path.read_text(encoding="utf-8")

    return run


def ready_service(indexed=4, newly_embedded=4):
    """Return a stand-in service that reports a healthy cache."""
    service = MagicMock()
    service.refresh.return_value = newly_embedded
    service.__len__.return_value = indexed

    return service


def test_the_recording_entry_point_is_handed_to_the_cli(startup):
    """Mode 2 learns, so the CLI drives record_error."""
    service = ready_service(indexed=7)

    run_cli, _ = startup(Mock(return_value=service))

    kwargs = run_cli.call_args.kwargs
    assert kwargs["record_fault_fn"] is service.record_error
    # The size callable reports how much was searched, nothing more.
    assert kwargs["log_size_fn"]() == 7


def test_nothing_is_recorded_during_startup(startup):
    """Startup only prepares the history; it never invents a fault."""
    service = ready_service()

    startup(Mock(return_value=service))

    service.record_error.assert_not_called()


def test_the_model_is_warmed_during_startup(startup):
    service = ready_service()

    startup(Mock(return_value=service))

    service.warm_up.assert_called_once_with()


def test_warm_up_happens_after_the_history_is_prepared(startup):
    """Refreshing first means a cold cache does not load the model twice."""
    service = ready_service()
    order = []
    service.refresh.side_effect = lambda: order.append("refresh") or 4
    service.warm_up.side_effect = lambda: order.append("warm_up")

    startup(Mock(return_value=service))

    assert order == ["refresh", "warm_up"]


def test_the_service_is_built_once_and_reused(startup):
    service = ready_service()
    factory = Mock(return_value=service)

    startup(factory)

    factory.assert_called_once_with()
    service.refresh.assert_called_once_with()


def test_the_cache_is_refreshed_at_startup(startup):
    service = ready_service(indexed=9, newly_embedded=9)

    _, log_text = startup(Mock(return_value=service))

    assert "Semantic log search preparation started." in log_text
    assert (
        "Semantic log search is ready with 9 indexed fault records "
        "(9 newly embedded) in "
        in log_text
    )


def test_a_warm_cache_embeds_nothing_new(startup):
    service = ready_service(indexed=9, newly_embedded=0)

    _, log_text = startup(Mock(return_value=service))

    assert "9 indexed fault records (0 newly embedded)" in log_text


def test_the_service_is_closed_when_the_application_exits(startup):
    service = ready_service()

    startup(Mock(return_value=service))

    service.close.assert_called_once_with()


def test_the_service_is_closed_even_if_the_cli_raises(monkeypatch, tmp_path):
    service = ready_service()
    log_path = tmp_path / "autocomplete.log"

    monkeypatch.setattr(
        application_main,
        "configure_logging",
        lambda: configure_logging(log_path),
    )
    monkeypatch.setattr(
        application_main,
        "initialize_corpus",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(application_main, "set_corpus_index", Mock())
    monkeypatch.setattr(
        application_main,
        "LogSearchService",
        Mock(return_value=service),
    )
    monkeypatch.setattr(
        application_main,
        "run_cli",
        Mock(side_effect=RuntimeError("cli exploded")),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "Archive.zip"])

    with pytest.raises(RuntimeError, match="cli exploded"):
        application_main.main()

    service.close.assert_called_once_with()


def test_a_failed_setup_leaves_regular_autocomplete_working(startup):
    factory = Mock(side_effect=RuntimeError("model could not be loaded"))

    run_cli, log_text = startup(factory)

    run_cli.assert_called_once_with()
    assert (
        "Semantic Log Search is unavailable because its preparation "
        "failed: model could not be loaded"
        in log_text
    )
    assert "Traceback" in log_text
    assert "The autocomplete system is ready for searches." in log_text


def test_a_failed_refresh_also_degrades_gracefully(startup):
    service = MagicMock()
    service.refresh.side_effect = RuntimeError("cache is corrupted")

    run_cli, log_text = startup(Mock(return_value=service))

    run_cli.assert_called_once_with()
    assert "cache is corrupted" in log_text


# ----------------------------------------------------------------------
# The old Gemini wiring must be gone from the runtime path
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module_path", ["main.py", "cli.py"])
def test_the_runtime_path_no_longer_imports_gemini(module_path):
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("google") for name in imported)
    assert "src.semantic.embedding_provider" not in imported
    assert "src.semantic.index_builder" not in imported
    assert "src.semantic.search" not in imported


@pytest.mark.parametrize("module_path", ["main.py", "cli.py"])
def test_the_old_corpus_embedding_cache_is_not_referenced(module_path):
    source = Path(module_path).read_text(encoding="utf-8")

    assert "embedding_cache_test" not in source
    assert "GEMINI_API_KEY" not in source


def test_the_gemini_source_files_are_still_present():
    """Removed from the runtime path, but kept in the repository."""
    assert Path("src/semantic/embedding_provider.py").is_file()
    assert Path("src/semantic/index_builder.py").is_file()
    assert Path("src/semantic/search.py").is_file()


def test_record_error_remains_available_for_the_satellite_flow():
    """Mode 2 does not use it, but the automatic flow will."""
    from src.logsearch.log_index import LogSearchService, record_error

    assert callable(record_error)
    assert callable(LogSearchService.record_error)


# ----------------------------------------------------------------------
# Startup performance metrics
# ----------------------------------------------------------------------


def test_startup_prints_the_cache_metrics(startup, capsys):
    service = ready_service(indexed=1200, newly_embedded=0)

    startup(Mock(return_value=service))

    output = capsys.readouterr().out
    assert "Semantic Log Search ready:" in output
    assert "  Historical faults: 1200" in output
    assert "  Newly embedded: 0" in output
    assert re.search(r"  Initialization time: \d+\.\d{3} sec", output)


def test_startup_metrics_show_work_done_on_a_cold_cache(startup, capsys):
    service = ready_service(indexed=1200, newly_embedded=1200)

    startup(Mock(return_value=service))

    output = capsys.readouterr().out
    assert "  Historical faults: 1200" in output
    assert "  Newly embedded: 1200" in output


def test_startup_logs_a_non_negative_initialization_time(startup):
    service = ready_service(indexed=9, newly_embedded=0)

    _, log_text = startup(Mock(return_value=service))

    match = re.search(
        r"9 indexed fault records \(0 newly embedded\) in "
        r"(\d+\.\d{3}) seconds\.",
        log_text,
    )
    assert match
    assert float(match.group(1)) >= 0.0


def test_no_metrics_are_printed_when_setup_fails(startup, capsys):
    startup(Mock(side_effect=RuntimeError("model could not be loaded")))

    assert "Semantic Log Search ready:" not in capsys.readouterr().out


def test_a_failed_warm_up_degrades_gracefully(startup):
    """A model that will not load must not take Regular Autocomplete down."""
    service = MagicMock()
    service.refresh.return_value = 0
    service.warm_up.side_effect = RuntimeError("model could not be loaded")

    run_cli, log_text = startup(Mock(return_value=service))

    run_cli.assert_called_once_with()
    assert "model could not be loaded" in log_text
    assert "The autocomplete system is ready for searches." in log_text
