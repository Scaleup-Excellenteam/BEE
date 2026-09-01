"""Tests for incremental semantic search over historical fault logs.

Every test injects a deterministic stub embedder, so nothing loads the
real model, reaches the network, or needs an API key.
"""

import ast
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.logsearch.log_index import LogSearchService
from src.semantic.contracts import SemanticResult


HISTORY = """2026-09-01 10:00:00 | INFO | Application started.
2026-09-01 10:00:05 | ERROR | Database connection refused.
2026-09-01 10:00:07 | INFO | Search completed successfully.
2026-09-01 10:00:09 | WARNING | Battery temperature is high.
2026-09-01 10:00:12 | ERROR | Telemetry packet checksum mismatch.
2026-09-01 10:00:15 | INFO | The system is ready.
2026-09-01 10:00:18 | CRITICAL | Attitude control lost.
"""


def letter_vector(text):
    """A deterministic stand-in for a sentence embedding model."""
    counts = Counter(character for character in text.lower() if character.isalpha())

    return [float(counts.get(chr(97 + index), 0)) + 0.1 for index in range(26)]


def stub_embedder(text):
    return letter_vector(text)


def stub_batch_embedder(texts):
    return [letter_vector(text) for text in texts]


@pytest.fixture
def log_path(tmp_path):
    """Write the sample history to a log file and return its path."""
    path = tmp_path / "logs" / "autocomplete.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HISTORY, encoding="utf-8")

    return path


@pytest.fixture
def make_service(tmp_path, log_path):
    """Build a service wired to the sample log and a temporary cache."""
    created = []

    def build(batch_embedder=None, embedder=None):
        service = LogSearchService(
            log_path=str(log_path),
            cache_path=str(tmp_path / "log_embedding_cache"),
            embedder=embedder or stub_embedder,
            batch_embedder=batch_embedder or stub_batch_embedder,
        )
        created.append(service)

        return service

    yield build

    for service in created:
        service.close()


def append_line(path, level, message):
    """Append one entry to the log the way logging would."""
    with Path(path).open("a", encoding="utf-8") as log_file:
        log_file.write(f"2026-09-01 11:00:00 | {level} | {message}\n")


# ----------------------------------------------------------------------
# First startup: build the cache from history
# ----------------------------------------------------------------------


def test_first_startup_builds_the_cache_from_fault_history(make_service):
    service = make_service()

    added = service.refresh()

    assert added == 4
    assert len(service) == 4


def test_only_fault_levels_are_indexed(make_service):
    service = make_service()
    service.refresh()

    indexed = [result.sentence for result in service.search_similar_logs("x", k=10)]

    assert sorted(indexed) == sorted(
        [
            "Database connection refused.",
            "Battery temperature is high.",
            "Telemetry packet checksum mismatch.",
            "Attitude control lost.",
        ]
    )
    assert "Application started." not in indexed
    assert "The system is ready." not in indexed


def test_the_original_message_source_and_offset_are_preserved(make_service):
    service = make_service()
    service.refresh()

    results = service.search_similar_logs("Database connection refused.", k=1)

    assert results[0].sentence == "Database connection refused."
    assert results[0].source_text == "autocomplete.log"
    assert results[0].offset == 2


def test_only_fault_messages_are_sent_to_the_embedder(make_service):
    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service = make_service(batch_embedder=batch_embedder)

    service.refresh()

    embedded = [text for call in batch_embedder.call_args_list for text in call.args[0]]
    assert len(embedded) == 4
    assert not any("Application started" in text for text in embedded)


def test_a_traceback_is_indexed_with_its_error(make_service, log_path):
    append_line(log_path, "ERROR", "Corpus preparation failed: boom")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("Traceback (most recent call last):\n")
        log_file.write("RuntimeError: boom\n")

    service = make_service()
    service.refresh()

    results = service.search_similar_logs("Corpus preparation failed", k=1)

    assert "RuntimeError: boom" in results[0].sentence


# ----------------------------------------------------------------------
# Later startups: incremental only
# ----------------------------------------------------------------------


def test_restarting_without_new_logs_embeds_nothing(make_service):
    first = make_service()
    first.refresh()
    first.close()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    second = make_service(batch_embedder=batch_embedder)

    assert second.refresh() == 0
    assert len(second) == 4
    batch_embedder.assert_not_called()


def test_restarting_embeds_only_the_new_records(make_service, log_path):
    first = make_service()
    first.refresh()
    first.close()

    append_line(log_path, "ERROR", "Solar panel deployment failed.")
    append_line(log_path, "INFO", "Routine status ping.")

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    second = make_service(batch_embedder=batch_embedder)

    assert second.refresh() == 1
    assert len(second) == 5

    embedded = [text for call in batch_embedder.call_args_list for text in call.args[0]]
    assert embedded == ["Solar panel deployment failed."]


def test_the_cache_survives_several_restarts(make_service, log_path):
    for index in range(3):
        service = make_service()
        service.refresh()
        service.close()
        append_line(log_path, "ERROR", f"Fault number {index}.")

    final = make_service()
    final.refresh()

    assert len(final) == 7


# ----------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------


def test_search_returns_semantic_results(make_service):
    service = make_service()
    service.refresh()

    results = service.search_similar_logs("battery temperature", k=5)

    assert results
    assert all(isinstance(result, SemanticResult) for result in results)
    assert results[0].sentence == "Battery temperature is high."


def test_search_returns_at_most_k_results(make_service):
    service = make_service()
    service.refresh()

    assert len(service.search_similar_logs("failure", k=2)) == 2
    assert len(service.search_similar_logs("failure", k=5)) == 4


def test_results_are_ordered_by_descending_similarity(make_service):
    service = make_service()
    service.refresh()

    similarities = [
        result.similarity
        for result in service.search_similar_logs("battery temperature", k=5)
    ]

    assert similarities == sorted(similarities, reverse=True)


def test_searching_an_empty_cache_returns_nothing(make_service):
    service = make_service()

    assert service.search_similar_logs("anything") == []


def test_searching_never_writes_to_the_cache(make_service):
    service = make_service()
    service.refresh()

    before = len(service)
    service.search_similar_logs("database", k=5)

    assert len(service) == before


# ----------------------------------------------------------------------
# The new-error flow: search BEFORE append
# ----------------------------------------------------------------------


def test_a_new_error_is_never_matched_against_itself(make_service):
    service = make_service()
    service.refresh()

    text = "Completely unprecedented gyroscope desynchronisation."
    results = service.record_error(text)

    assert all(result.sentence != text for result in results)


def test_a_new_error_is_appended_to_the_log(make_service, log_path):
    service = make_service()
    service.refresh()

    service.record_error("Solar panel deployment failed.")

    assert "| ERROR | Solar panel deployment failed." in log_path.read_text(
        encoding="utf-8"
    )


def test_a_new_error_is_appended_to_the_cache(make_service):
    service = make_service()
    service.refresh()

    service.record_error("Solar panel deployment failed.")

    assert len(service) == 5


def test_a_recorded_error_becomes_searchable_afterwards(make_service):
    service = make_service()
    service.refresh()

    text = "Solar panel deployment failed."
    service.record_error(text)

    assert any(
        result.sentence == text
        for result in service.search_similar_logs(text, k=5)
    )


def test_recording_returns_the_top_five_historical_faults(make_service, log_path):
    for index in range(8):
        append_line(log_path, "ERROR", f"Historical fault number {index}.")

    service = make_service()
    service.refresh()

    results = service.record_error("Historical fault number nine.")

    assert len(results) == 5


def test_recording_repeatedly_keeps_the_store_usable(make_service):
    service = make_service()
    service.refresh()

    for index in range(3):
        service.record_error(f"Repeated fault {index}.")

    assert len(service) == 7
    assert service.search_similar_logs("Repeated fault 1.", k=1)


def test_recording_does_not_re_embed_the_history(make_service):
    service = make_service()
    service.refresh()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service.batch_embedder = batch_embedder

    service.record_error("Solar panel deployment failed.")

    embedded = [text for call in batch_embedder.call_args_list for text in call.args[0]]
    assert embedded == ["Solar panel deployment failed."]


def test_a_non_fault_level_is_rejected(make_service):
    service = make_service()
    service.refresh()

    with pytest.raises(ValueError, match="not a fault level"):
        service.record_error("Routine ping.", level="INFO")


def test_warning_and_critical_can_be_recorded(make_service):
    service = make_service()
    service.refresh()

    service.record_error("Battery cell imbalance detected.", level="WARNING")
    service.record_error("Reaction wheel seized.", level="CRITICAL")

    assert len(service) == 6


# ----------------------------------------------------------------------
# Independence from Gemini and from Part A
# ----------------------------------------------------------------------


def test_the_whole_flow_works_without_a_gemini_api_key(monkeypatch, make_service):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    service = make_service()
    service.refresh()
    results = service.record_error("Attitude control degraded.")

    assert results
    assert len(service) == 5


@pytest.mark.parametrize(
    "module_path",
    [
        "src/logsearch/log_index.py",
        "src/logsearch/log_parser.py",
        "src/semantic/local_provider.py",
    ],
)
def test_the_log_search_path_never_imports_gemini(module_path):
    """The runtime path must not reach the API provider at all."""
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("google") for name in imported)
    assert "src.semantic.embedding_provider" not in imported


def test_part_a_autocomplete_is_untouched_by_the_log_engine():
    """Importing the log engine must not disturb Part A behaviour."""
    import autocomplete
    from src.corpus.index import CorpusIndex
    from src.models import AutoCompleteData, SentenceRecord

    autocomplete.set_corpus_index(
        CorpusIndex(
            [
                SentenceRecord(
                    original_sentence="Hello World",
                    normalized_sentence="hello world",
                    source_text="greetings.txt",
                    offset=1,
                )
            ]
        )
    )

    assert autocomplete.get_best_k_completions("HELLO!") == [
        AutoCompleteData(
            completed_sentence="Hello World",
            source_text="greetings.txt",
            offset=1,
            score=10,
        )
    ]
