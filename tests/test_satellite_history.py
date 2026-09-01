"""Tests for the unified satellite fault history.

The history is built from two sources that must behave as one:

    data/historical_satellite_faults.log   imported once, at first start
    logs/satellite_faults.log              grown at runtime by record_error

Every test injects a deterministic stub embedder, so nothing loads the
real model or reaches the network.
"""

from collections import Counter
from unittest.mock import Mock

import pytest

from src.logsearch import log_index
from src.logsearch.log_index import LogSearchService
from src.semantic.embedding_cache import CacheError


DATASET = """2024-03-11 04:12:07 | ERROR | Battery voltage dropped below safe level
2024-03-11 06:48:52 | CRITICAL | Communication link lost
2024-03-12 22:05:19 | WARNING | Reaction wheel temperature too high
2024-04-02 09:31:44 | INFO | Routine pass completed
"""

LIVE_FAULT = "Battery power critically low"


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
def satellite(tmp_path):
    """Build services that share one dataset, history and cache."""
    dataset_path = tmp_path / "data" / "historical_satellite_faults.log"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(DATASET, encoding="utf-8")

    history_path = tmp_path / "logs" / "satellite_faults.log"
    cache_path = str(tmp_path / "satellite_fault_cache")
    created = []

    def build(batch_embedder=None, log_path=None, dataset=dataset_path):
        service = LogSearchService(
            log_path=str(log_path or history_path),
            cache_path=cache_path,
            dataset_path=str(dataset) if dataset else None,
            embedder=stub_embedder,
            batch_embedder=batch_embedder or stub_batch_embedder,
        )
        created.append(service)

        return service

    yield build, history_path, cache_path

    for service in created:
        service.close()


def stored_sentences(service):
    """Return every sentence in the cache, ignoring the threshold."""
    previous = service.min_similarity
    service.min_similarity = 0.0
    try:
        return {
            result.sentence
            for result in service.search_similar_logs("x", k=50)
        }
    finally:
        service.min_similarity = previous


# ----------------------------------------------------------------------
# Bootstrap from the prepared dataset
# ----------------------------------------------------------------------


def test_the_prepared_dataset_is_imported_on_first_start(satellite):
    build, history_path, _ = satellite
    service = build()

    embedded = service.refresh()

    assert history_path.is_file()
    # Three faults; the INFO line is copied but never indexed.
    assert embedded == 3
    assert len(service) == 3


def test_the_prepared_faults_are_searchable(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    assert stored_sentences(service) == {
        "Battery voltage dropped below safe level",
        "Communication link lost",
        "Reaction wheel temperature too high",
    }


def test_the_dataset_is_imported_only_once(satellite):
    build, history_path, _ = satellite
    build().refresh()
    first_history = history_path.read_text(encoding="utf-8")

    second = build()
    embedded_again = second.refresh()

    assert embedded_again == 0
    assert len(second) == 3
    assert history_path.read_text(encoding="utf-8") == first_history


def test_a_restart_re_embeds_nothing(satellite):
    build, _, _ = satellite
    build().refresh()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    restarted = build(batch_embedder=batch_embedder)
    restarted.refresh()

    batch_embedder.assert_not_called()
    assert len(restarted) == 3


def test_a_missing_dataset_is_not_an_error(satellite, tmp_path):
    build, _, _ = satellite
    service = build(dataset=tmp_path / "nothing_here.log")

    assert service.refresh() == 0
    assert service.search_similar_logs("anything") == []


def test_the_dataset_never_overwrites_an_existing_history(satellite):
    build, history_path, _ = satellite
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "2026-01-01 00:00:00 | ERROR | Pre-existing fault\n",
        encoding="utf-8",
    )

    service = build()
    service.refresh()

    history = history_path.read_text(encoding="utf-8")
    assert "Pre-existing fault" in history
    assert "Communication link lost" not in history
    assert len(service) == 1


def test_the_real_repository_dataset_is_present_and_parses():
    """The shipped dataset must actually be loadable."""
    from src.logsearch.log_parser import parse_fault_records

    records = parse_fault_records(log_index.BOOTSTRAP_DATASET_PATH)

    assert len(records) >= 3
    assert {record.level for record in records} <= {
        "ERROR",
        "WARNING",
        "CRITICAL",
    }


# ----------------------------------------------------------------------
# Live incremental learning
# ----------------------------------------------------------------------


def test_a_live_fault_joins_the_prepared_history(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    service.record_error(LIVE_FAULT)

    assert len(service) == 4
    sentences = stored_sentences(service)
    assert "Battery voltage dropped below safe level" in sentences
    assert LIVE_FAULT in sentences


def test_a_live_fault_is_appended_to_the_history_file(satellite):
    build, history_path, _ = satellite
    service = build()
    service.refresh()

    service.record_error(LIVE_FAULT)

    assert f"| ERROR | {LIVE_FAULT}" in history_path.read_text(
        encoding="utf-8"
    )


def test_a_live_fault_only_embeds_itself(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service.batch_embedder = batch_embedder
    service.record_error(LIVE_FAULT)

    embedded = [
        text
        for call in batch_embedder.call_args_list
        for text in call.args[0]
    ]
    assert embedded == [LIVE_FAULT]


def test_a_live_fault_is_searched_before_it_is_inserted(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    results = service.record_error(LIVE_FAULT)

    assert all(result.sentence != LIVE_FAULT for result in results)


def test_live_faults_survive_a_restart(satellite):
    build, _, _ = satellite
    first = build()
    first.refresh()
    first.record_error(LIVE_FAULT)
    first.close()

    restarted = build()

    assert restarted.refresh() == 0
    assert len(restarted) == 4
    assert LIVE_FAULT in stored_sentences(restarted)


def test_several_live_faults_accumulate(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    service.record_error("Solar array output degraded")
    service.record_error("Transponder failed to lock")
    service.record_error("Gyroscope drift detected", level="WARNING")

    assert len(service) == 6


def test_both_histories_are_searched_together(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()
    service.record_error(LIVE_FAULT)

    results = service.search_similar_logs("Battery voltage dropped", k=5)

    sentences = [result.sentence for result in results]
    assert "Battery voltage dropped below safe level" in sentences
    assert LIVE_FAULT in sentences


def test_the_threshold_still_applies_to_the_unified_history(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    assert service.min_similarity == 0.35
    assert log_index.MIN_LOG_SIMILARITY == 0.35


# ----------------------------------------------------------------------
# The satellite history is not the application log
# ----------------------------------------------------------------------


def test_the_default_history_is_not_the_application_log():
    assert log_index.DEFAULT_LOG_PATH == "logs/satellite_faults.log"
    assert "autocomplete" not in log_index.DEFAULT_LOG_PATH
    assert "autocomplete" not in log_index.DEFAULT_CACHE_PATH
    assert log_index.BOOTSTRAP_DATASET_PATH == (
        "data/historical_satellite_faults.log"
    )


def test_the_application_log_is_never_read(satellite, monkeypatch):
    build, _, _ = satellite
    read_paths = []
    original = log_index.parse_fault_records

    def spy(path, *arguments, **keywords):
        read_paths.append(str(path))
        return original(path, *arguments, **keywords)

    monkeypatch.setattr(log_index, "parse_fault_records", spy)

    service = build()
    service.refresh()
    service.search_similar_logs("anything")
    service.record_error(LIVE_FAULT)

    assert read_paths
    assert not any("autocomplete" in path for path in read_paths)


# ----------------------------------------------------------------------
# A wrong or stale cache must not be reused silently
# ----------------------------------------------------------------------


def test_a_cache_built_from_another_log_is_rejected(satellite, tmp_path):
    build, _, cache_path = satellite
    service = build()
    service.refresh()
    service.close()

    other_history = tmp_path / "logs" / "some_other.log"
    other_history.write_text(
        "2026-01-01 00:00:00 | ERROR | Unrelated fault\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="was built from"):
        build(log_path=other_history, dataset=None)


def test_a_cache_ahead_of_its_history_is_rejected(satellite):
    build, history_path, _ = satellite
    service = build()
    service.refresh()
    service.close()

    # The history was replaced by a shorter one, so the cache still
    # claims records at lines that no longer exist.
    history_path.write_text(
        "2026-01-01 00:00:00 | ERROR | Only one fault now\n",
        encoding="utf-8",
    )

    restarted = build()
    with pytest.raises(CacheError, match="no longer has"):
        restarted.refresh()


# ----------------------------------------------------------------------
# Mode 2 learns: the exact flow the CLI now drives
# ----------------------------------------------------------------------


def test_an_entered_fault_is_persisted_after_being_searched(satellite):
    build, history_path, _ = satellite
    service = build()
    service.refresh()
    before = len(service)

    service.record_error("link lost")

    assert len(service) == before + 1
    assert "| ERROR | link lost" in history_path.read_text(encoding="utf-8")


def test_the_next_entry_can_find_the_previous_one(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    service.record_error("link lost")
    results = service.record_error("Communication lost")

    sentences = [result.sentence for result in results]
    assert "link lost" in sentences
    assert "Communication link lost" in sentences


def test_an_entry_can_never_match_itself(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    for message in ["link lost", "link lost again", "link lost once more"]:
        results = service.record_error(message)
        assert all(result.sentence != message for result in results)


def test_the_history_grows_by_one_per_entry(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()
    sizes = [len(service)]

    for message in ["link lost", "Communication lost", "Battery low"]:
        service.record_error(message)
        sizes.append(len(service))

    assert sizes == [3, 4, 5, 6]


def test_entries_are_searched_against_the_history_before_insertion(satellite):
    """Each entry sees exactly the history that existed before it."""
    build, _, _ = satellite
    service = build()
    service.refresh()
    service.min_similarity = 0.0

    seen = []
    for message in ["first entry", "second entry", "third entry"]:
        seen.append(len(service.search_similar_logs(message, k=50)))
        service.record_error(message)

    assert seen == [3, 4, 5]


def test_entries_survive_a_restart(satellite):
    build, _, _ = satellite
    first = build()
    first.refresh()
    first.record_error("link lost")
    first.record_error("Communication lost")
    first.close()

    restarted = build()

    assert restarted.refresh() == 0
    assert len(restarted) == 5
    sentences = stored_sentences(restarted)
    assert "link lost" in sentences
    assert "Communication lost" in sentences


def test_a_blank_fault_is_refused(satellite):
    build, _, _ = satellite
    service = build()
    service.refresh()

    for blank in ["", "   ", "\t"]:
        with pytest.raises(ValueError, match="must have a message"):
            service.record_error(blank)

    assert len(service) == 3
