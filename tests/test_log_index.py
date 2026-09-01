"""Tests for the satellite fault memory service.

Every test injects a deterministic stub embedder, so nothing loads the
real model, reaches the network, or needs an API key.
"""

from collections import Counter
from unittest.mock import Mock

import pytest

from src.logsearch import log_index
from src.logsearch.incidents import IncidentMatch
from src.logsearch.log_index import LogSearchService
from src.semantic.embedding_cache import CacheError


DATASET = """2024-03-11 04:12:07 | ERROR | Battery voltage dropped below safe level
2024-03-11 06:48:52 | CRITICAL | Communication link lost
2024-03-12 22:05:19 | WARNING | Reaction wheel temperature too high
2024-04-02 09:31:44 | INFO | Routine pass completed
"""


def letter_vector(text):
    """A deterministic stand-in for a sentence embedding model."""
    counts = Counter(
        character for character in text.lower() if character.isalpha()
    )

    return [float(counts.get(chr(97 + index), 0)) + 0.1 for index in range(26)]


def stub_embedder(text):
    return letter_vector(text)


def stub_batch_embedder(texts):
    return [letter_vector(text) for text in texts]


@pytest.fixture
def make_service(tmp_path):
    """Build services sharing one dataset, incident history and cache."""
    dataset_path = tmp_path / "data" / "historical_satellite_faults.log"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(DATASET, encoding="utf-8")

    incidents_path = tmp_path / "logs" / "satellite_incidents.jsonl"
    cache_path = str(tmp_path / "satellite_fault_cache")
    created = []

    def build(batch_embedder=None, dataset=dataset_path, clock=None):
        service = LogSearchService(
            incidents_path=str(incidents_path),
            cache_path=cache_path,
            dataset_path=str(dataset) if dataset else None,
            legacy_log_path=None,
            embedder=stub_embedder,
            batch_embedder=batch_embedder or stub_batch_embedder,
            warm_up_fn=Mock(),
            now_fn=clock or (lambda: "2026-09-01 12:00:00"),
        )
        created.append(service)

        return service

    yield build, incidents_path, dataset_path, cache_path

    for service in created:
        service.close()


def ready(make_service, **keywords):
    """Return a service with its history bootstrapped and embedded."""
    build = make_service[0]
    service = build(**keywords)
    service.refresh()

    return service


# ----------------------------------------------------------------------
# Bootstrap and migration
# ----------------------------------------------------------------------


def test_the_prepared_dataset_becomes_structured_incidents(make_service):
    service = ready(make_service)

    # Three faults; the INFO line is never a fault.
    assert len(service) == 3
    assert [incident.message for incident in service.incidents] == [
        "Battery voltage dropped below safe level",
        "Communication link lost",
        "Reaction wheel temperature too high",
    ]


def test_migration_uses_honest_defaults(make_service):
    service = ready(make_service)
    first = service.incidents[0]

    assert first.incident_id == 1
    assert first.subsystem == "UNKNOWN"
    assert first.severity == "ERROR"
    assert first.error_code is None
    assert first.count == 1
    assert first.first_seen == "2024-03-11 04:12:07"
    assert first.last_seen == first.first_seen


def test_migration_keeps_the_parsed_severity(make_service):
    service = ready(make_service)

    assert [incident.severity for incident in service.incidents] == [
        "ERROR",
        "CRITICAL",
        "WARNING",
    ]


def test_the_dataset_is_migrated_only_once(make_service):
    build, _, _, _ = make_service
    ready(make_service)

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    second = build(batch_embedder=batch_embedder)

    assert second.refresh() == 0
    assert len(second) == 3
    batch_embedder.assert_not_called()


def test_the_dataset_file_is_never_modified(make_service):
    _, _, dataset_path, _ = make_service
    ready(make_service)

    assert dataset_path.read_text(encoding="utf-8") == DATASET


def test_a_missing_dataset_is_not_an_error(make_service, tmp_path):
    service = ready(make_service, dataset=tmp_path / "absent.log")

    assert len(service) == 0
    assert service.search_similar_logs("anything") == []


def test_a_legacy_fault_log_is_migrated_when_present(tmp_path):
    legacy = tmp_path / "logs" / "satellite_faults.log"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "2026-01-01 00:00:00 | ERROR | A previously recorded fault\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "data" / "prepared.log"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(DATASET, encoding="utf-8")

    service = LogSearchService(
        incidents_path=str(tmp_path / "incidents.jsonl"),
        cache_path=str(tmp_path / "cache"),
        dataset_path=str(dataset),
        legacy_log_path=str(legacy),
        embedder=stub_embedder,
        batch_embedder=stub_batch_embedder,
        warm_up_fn=Mock(),
    )
    service.refresh()

    # The existing history wins: it already contains the dataset plus
    # anything recorded after it.
    assert len(service) == 1
    assert service.incidents[0].message == "A previously recorded fault"
    service.close()


def test_the_real_repository_dataset_is_present_and_parses():
    from src.logsearch.log_parser import parse_fault_records

    records = parse_fault_records(log_index.BOOTSTRAP_DATASET_PATH)

    assert len(records) >= 3
    assert {record.level for record in records} <= {
        "ERROR",
        "WARNING",
        "CRITICAL",
    }


# ----------------------------------------------------------------------
# Recording a genuinely new fault
# ----------------------------------------------------------------------


def test_a_new_fault_becomes_a_new_incident(make_service):
    service = ready(make_service)

    service.record_error(
        "Payload instrument returned corrupted data",
        subsystem="PAYLOAD",
        severity="WARNING",
    )

    assert len(service) == 4
    newest = service.incidents[-1]
    assert newest.subsystem == "PAYLOAD"
    assert newest.severity == "WARNING"
    assert newest.count == 1
    assert newest.first_seen == newest.last_seen == "2026-09-01 12:00:00"


def test_a_new_fault_defaults_to_unknown_subsystem(make_service):
    service = ready(make_service)

    service.record_error("Something odd happened in orbit")

    assert service.incidents[-1].subsystem == "UNKNOWN"
    assert service.incidents[-1].severity == "ERROR"


def test_a_new_fault_records_its_optional_metadata(make_service):
    service = ready(make_service)

    service.record_error(
        "Battery voltage dropped below safe level",
        subsystem="POWER",
        severity="CRITICAL",
        error_code="PWR-LOW-VOLTAGE",
        previous_action="Switched to reserve cells",
        outcome="Voltage recovered",
    )

    newest = service.incidents[-1]
    assert newest.error_code == "PWR-LOW-VOLTAGE"
    assert newest.previous_action == "Switched to reserve cells"
    assert newest.outcome == "Voltage recovered"


def test_a_distinct_fault_creates_another_embedding(make_service):
    service = ready(make_service)
    vectors_before = len(service._store)

    service.record_error("Battery temperature sensor failure", subsystem="POWER")

    assert len(service._store) == vectors_before + 1
    assert len(service._store) == len(service)


def test_two_power_faults_are_not_merged_just_for_sharing_a_subsystem(
    make_service,
):
    service = ready(make_service)

    service.record_error("Battery voltage low", subsystem="POWER")
    service.record_error("Battery temperature sensor failure", subsystem="POWER")

    assert len(service) == 5


def test_a_fault_is_searched_before_it_is_recorded(make_service):
    service = ready(make_service)

    text = "Propulsion valve stuck open unexpectedly"
    outcome = service.record_error(text, subsystem="UNKNOWN")

    # The fault itself was not in the history when the search ran.
    assert all(match.message != text for match in outcome.matches)
    assert outcome.stored is True
    assert len(service) == 4


def test_a_repeat_still_only_sees_earlier_incidents(make_service):
    """Even a duplicate is matched against history, never against itself."""
    service = ready(make_service)
    service.record_error("Solar array degraded", subsystem="POWER")

    outcome = service.record_error("Solar array degraded", subsystem="POWER")

    assert len(service) == 4
    assert outcome.deduplicated is True
    assert service.incidents[-1].count == 2
    # The one it merged into is a PREVIOUS incident, ranked normally.
    assert all(match.incident.incident_id <= 4 for match in outcome.matches)


def test_a_blank_fault_is_refused(make_service):
    service = ready(make_service)

    for blank in ["", "   ", "\t"]:
        with pytest.raises(ValueError, match="must have a message"):
            service.record_error(blank)

    assert len(service) == 3


def test_an_unknown_severity_is_refused(make_service):
    service = ready(make_service)

    with pytest.raises(ValueError, match="not a fault severity"):
        service.record_error("A fault", severity="FATAL")


# ----------------------------------------------------------------------
# Recording a repeat
# ----------------------------------------------------------------------


def test_an_identical_fault_deduplicates(make_service):
    service = ready(make_service)
    service.record_error("Communication link lost", subsystem="COMMUNICATION")
    before = len(service)

    service.record_error("Communication link lost", subsystem="COMMUNICATION")

    assert len(service) == before


def test_a_repeat_increments_the_count(make_service):
    service = ready(make_service)
    service.record_error("Communication link lost", subsystem="COMMUNICATION")
    incident = service.incidents[-1]

    service.record_error("Communication link lost", subsystem="COMMUNICATION")

    assert incident.count == 2


def test_a_repeat_updates_last_seen_but_not_first_seen(make_service):
    build, _, _, _ = make_service
    times = iter(["2026-05-01 00:00:00", "2026-08-20 00:00:00"])
    service = build(clock=lambda: next(times))
    service.refresh()

    service.record_error("Communication link lost", subsystem="COMMUNICATION")
    incident = service.incidents[-1]
    service.record_error("Communication link lost", subsystem="COMMUNICATION")

    assert incident.first_seen == "2026-05-01 00:00:00"
    assert incident.last_seen == "2026-08-20 00:00:00"
    assert incident.count == 2


def test_a_repeat_creates_no_new_embedding(make_service):
    service = ready(make_service)
    service.record_error("Communication link lost", subsystem="COMMUNICATION")
    vectors_before = len(service._store)

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service.batch_embedder = batch_embedder
    service.record_error("Communication link lost", subsystem="COMMUNICATION")

    batch_embedder.assert_not_called()
    assert len(service._store) == vectors_before


def test_an_error_code_deduplicates_across_different_wording(make_service):
    service = ready(make_service)
    service.record_error(
        "Battery voltage dropped below safe level",
        subsystem="POWER",
        error_code="PWR-LOW-VOLTAGE",
    )
    before = len(service)

    service.record_error(
        "Bus voltage under limit",
        subsystem="POWER",
        error_code="PWR-LOW-VOLTAGE",
    )

    assert len(service) == before
    assert service.incidents[-1].count == 2


def test_the_same_error_code_in_another_subsystem_creates_an_incident(
    make_service,
):
    service = ready(make_service)
    service.record_error(
        "Battery voltage low",
        subsystem="POWER",
        error_code="SHARED-CODE",
    )
    before = len(service)

    service.record_error(
        "Battery voltage low",
        subsystem="THERMAL",
        error_code="SHARED-CODE",
    )

    assert len(service) == before + 1


def test_a_moderate_match_is_not_treated_as_a_repeat(make_service):
    """0.35 is enough to display, nowhere near enough to merge."""
    service = ready(make_service)
    service.dedup_threshold = 0.90
    before = len(service)

    # Related enough to be shown, nothing like identical.
    service.record_error("Battery cell imbalance", subsystem="UNKNOWN")

    assert len(service) == before + 1


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def test_search_returns_incident_matches(make_service):
    service = ready(make_service)

    results = service.search_similar_logs("Communication link lost")

    assert results
    assert all(isinstance(match, IncidentMatch) for match in results)
    assert results[0].message == "Communication link lost"
    assert results[0].subsystem == "UNKNOWN"
    assert results[0].count == 1


def test_search_never_records_anything(make_service):
    service = ready(make_service)

    service.search_similar_logs("Communication link lost")

    assert len(service) == 3
    assert all(incident.count == 1 for incident in service.incidents)


def test_results_are_ordered_by_descending_similarity(make_service):
    service = ready(make_service)

    similarities = [
        match.similarity
        for match in service.search_similar_logs("battery voltage", k=5)
    ]

    assert similarities == sorted(similarities, reverse=True)


def test_a_frequent_weak_match_never_outranks_a_strong_one(make_service):
    """Count is metadata; it must not touch the ordering."""
    service = ready(make_service)
    weak = service.incidents[2]
    weak.count = 900

    results = service.search_similar_logs("Communication link lost", k=5)

    assert results[0].message == "Communication link lost"
    assert results[0].count == 1
    assert results[0].similarity > results[-1].similarity


def test_a_critical_incident_never_outranks_a_better_match(make_service):
    """Severity is metadata; it must not touch the ordering either."""
    service = ready(make_service)
    for incident in service.incidents:
        incident.severity = "CRITICAL"
    service.incidents[1].severity = "WARNING"

    results = service.search_similar_logs("Communication link lost", k=5)

    assert results[0].message == "Communication link lost"


def test_weak_matches_are_filtered_out(make_service, monkeypatch):
    """Anything under the search threshold is not worth showing."""
    from src.semantic.contracts import SemanticResult

    service = ready(make_service)
    monkeypatch.setattr(
        log_index,
        "semantic_search",
        Mock(
            return_value=[
                SemanticResult("a", "s", 1, 0.81),
                SemanticResult("b", "s", 2, 0.41),
                SemanticResult("c", "s", 3, 0.34),
            ]
        ),
    )

    similarities = [
        match.similarity for match in service.search_similar_logs("anything")
    ]

    assert similarities == [0.81, 0.41]


def test_a_result_exactly_on_the_search_threshold_is_kept(
    make_service,
    monkeypatch,
):
    from src.semantic.contracts import SemanticResult

    service = ready(make_service)
    monkeypatch.setattr(
        log_index,
        "semantic_search",
        Mock(return_value=[SemanticResult("a", "s", 1, 0.35)]),
    )

    assert len(service.search_similar_logs("anything")) == 1


def test_the_thresholds_are_configured_as_expected(make_service):
    service = ready(make_service)

    assert service.min_similarity == 0.35
    assert log_index.MIN_LOG_SIMILARITY == 0.35
    assert service.dedup_threshold == 0.90


def test_searching_an_empty_history_returns_nothing(make_service, tmp_path):
    service = ready(make_service, dataset=tmp_path / "absent.log")

    assert service.search_similar_logs("anything") == []


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_the_history_file_is_the_canonical_record(make_service):
    _, incidents_path, _, _ = make_service
    ready(make_service)

    assert incidents_path.is_file()
    lines = incidents_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_metadata_survives_a_restart(make_service):
    build, _, _, _ = make_service
    first = ready(make_service)
    first.record_error(
        "Communication link lost",
        subsystem="COMMUNICATION",
        severity="CRITICAL",
        error_code="COM-LINK-LOST",
        previous_action="Switched to backup transponder",
        outcome="Link restored",
    )
    first.record_error(
        "Communication link lost",
        subsystem="COMMUNICATION",
        error_code="COM-LINK-LOST",
    )
    first.close()

    restarted = build()

    assert restarted.refresh() == 0
    assert len(restarted) == 4
    restored = restarted.incidents[-1]
    assert restored.count == 2
    assert restored.subsystem == "COMMUNICATION"
    assert restored.severity == "CRITICAL"
    assert restored.error_code == "COM-LINK-LOST"
    assert restored.previous_action == "Switched to backup transponder"
    assert restored.outcome == "Link restored"


def test_a_restart_reuses_the_embeddings(make_service):
    build, _, _, _ = make_service
    ready(make_service).close()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    restarted = build(batch_embedder=batch_embedder)

    assert restarted.refresh() == 0
    batch_embedder.assert_not_called()


def test_a_deleted_cache_is_rebuilt_from_the_history(make_service, tmp_path):
    import shutil

    build, _, _, cache_path = make_service
    first = ready(make_service)
    first.record_error("Solar array degraded", subsystem="POWER")
    first.close()

    shutil.rmtree(cache_path)
    rebuilt = build()

    # The history is the source of truth, so nothing was lost.
    assert rebuilt.refresh() == 4
    assert len(rebuilt) == 4


def test_a_cache_from_another_source_is_rejected(make_service, tmp_path):
    from src.semantic.contracts import EmbeddedSentence
    from src.semantic.embedding_cache import save_embeddings

    build, _, _, cache_path = make_service
    save_embeddings(
        [
            EmbeddedSentence(
                sentence="Something else entirely",
                source_text="autocomplete.log",
                offset=1,
                embedding=[1.0, 2.0],
            )
        ],
        cache_path,
    )

    with pytest.raises(CacheError, match="was built from"):
        build()


def test_a_cache_longer_than_the_history_is_rebuilt(make_service):
    """The history is canonical, so a stale cache recovers, not fails."""
    build, incidents_path, _, _ = make_service
    ready(make_service).close()

    # The history was truncated; the cache still holds three vectors.
    lines = incidents_path.read_text(encoding="utf-8").splitlines()
    incidents_path.write_text(lines[0] + "\n", encoding="utf-8")

    recovered = build()

    assert recovered.refresh() == 1
    assert len(recovered) == 1
    assert len(recovered._store) == 1


# ----------------------------------------------------------------------
# Warm-up
# ----------------------------------------------------------------------


def test_warm_up_loads_the_model(make_service):
    service = ready(make_service)

    service.warm_up()

    service.warm_up_fn.assert_called_once_with()


def test_warm_up_embeds_no_incident_text(make_service):
    build, _, _, _ = make_service
    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service = build(batch_embedder=batch_embedder)
    service.refresh()
    calls_after_refresh = batch_embedder.call_count

    service.warm_up()

    assert batch_embedder.call_count == calls_after_refresh


# ----------------------------------------------------------------------
# The satellite history is not the application log
# ----------------------------------------------------------------------


def test_the_default_history_is_not_the_application_log():
    from src.logsearch.incident_history import DEFAULT_INCIDENTS_PATH

    assert DEFAULT_INCIDENTS_PATH == "logs/satellite_incidents.jsonl"
    assert "autocomplete" not in DEFAULT_INCIDENTS_PATH
    assert "autocomplete" not in log_index.DEFAULT_CACHE_PATH
    assert "autocomplete" not in log_index.BOOTSTRAP_DATASET_PATH
    assert "autocomplete" not in log_index.LEGACY_LOG_PATH


def test_the_application_log_is_never_read(make_service, monkeypatch):
    read_paths = []
    original = log_index.parse_fault_records

    def spy(path, *arguments, **keywords):
        read_paths.append(str(path))
        return original(path, *arguments, **keywords)

    monkeypatch.setattr(log_index, "parse_fault_records", spy)

    service = ready(make_service)
    service.search_similar_logs("anything")
    service.record_error("A new fault")

    assert read_paths
    assert not any("autocomplete" in path for path in read_paths)


# ----------------------------------------------------------------------
# Guards carried over from the previous integration branch
#
# These behaviours were pinned before the incident model landed and are
# not covered by the incident tests above, so they are kept, adapted to
# the structured history.
# ----------------------------------------------------------------------


def test_search_returns_at_most_k_results(make_service):
    service = ready(make_service)
    service.min_similarity = 0.0

    assert len(service.search_similar_logs("failure", k=2)) == 2
    assert len(service.search_similar_logs("failure", k=5)) == 3


def test_a_traceback_is_indexed_with_its_error(make_service, tmp_path):
    """A logged exception stays searchable together with its traceback."""
    dataset = tmp_path / "with_traceback.log"
    dataset.write_text(
        "2026-01-01 00:00:00 | ERROR | Corpus preparation failed: boom\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: boom\n",
        encoding="utf-8",
    )
    service = ready(make_service, dataset=dataset)

    results = service.search_similar_logs("Corpus preparation failed", k=1)

    assert "RuntimeError: boom" in results[0].message


def test_a_recorded_fault_becomes_searchable_afterwards(make_service):
    service = ready(make_service)
    text = "Solar array deployment stalled midway"

    service.record_error(text, subsystem="POWER")

    assert any(
        match.message == text
        for match in service.search_similar_logs(text, k=5)
    )


def test_the_whole_flow_works_without_a_gemini_api_key(
    monkeypatch,
    make_service,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    service = ready(make_service)
    outcome = service.record_error("Attitude control degraded")

    assert outcome.stored is True
    assert len(service) == 4


@pytest.mark.parametrize(
    "module_path",
    [
        "src/logsearch/log_index.py",
        "src/logsearch/log_parser.py",
        "src/logsearch/incidents.py",
        "src/logsearch/incident_history.py",
        "src/logsearch/retention.py",
        "src/semantic/local_provider.py",
    ],
)
def test_the_log_search_path_never_imports_gemini(module_path):
    """The runtime path must not reach the API provider at all."""
    import ast
    from pathlib import Path

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
