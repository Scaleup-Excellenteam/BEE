"""Tests for upgrading a deployment written by the previous version.

The old layout on disk was:

    logs/satellite_faults.log     plain append-only fault log
    satellite_fault_cache/        vectors tagged "satellite_faults.log"

and there was no incident history at all.  Starting the new code on that
state must convert it automatically, without the operator deleting
anything and without losing a single learned fault.
"""

from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.logsearch.incident_history import IncidentHistory
from src.logsearch.log_index import LogSearchService
from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import CacheError, save_embeddings


# Six prepared faults plus three learned at runtime, exactly the shape a
# previous version would have left behind.
PREPARED = """2024-03-11 04:12:07 | ERROR | Battery voltage dropped below safe level
2024-03-11 06:48:52 | CRITICAL | Communication link lost
2024-03-12 22:05:19 | WARNING | Reaction wheel temperature too high
"""

LEARNED = """2026-08-01 09:00:00 | ERROR | Solar array deployment stalled midway
2026-08-02 11:30:00 | CRITICAL | Star tracker lost its attitude reference
2026-08-03 15:45:00 | WARNING | Downlink margin degraded during the pass
"""

LEGACY_LOG = PREPARED + LEARNED


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
def upgrade(tmp_path):
    """Lay out the old version's disk state and build services on it."""
    legacy_log = tmp_path / "logs" / "satellite_faults.log"
    legacy_log.parent.mkdir(parents=True, exist_ok=True)
    legacy_log.write_text(LEGACY_LOG, encoding="utf-8")

    dataset = tmp_path / "data" / "historical_satellite_faults.log"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(PREPARED, encoding="utf-8")

    incidents_path = tmp_path / "logs" / "satellite_incidents.jsonl"
    cache_path = str(tmp_path / "satellite_fault_cache")
    created = []

    def build(batch_embedder=None, max_incidents=100, legacy_log_path=None):
        service = LogSearchService(
            incidents_path=str(incidents_path),
            cache_path=cache_path,
            dataset_path=str(dataset),
            legacy_log_path=str(
                legacy_log if legacy_log_path is None else legacy_log_path
            ),
            max_incidents=max_incidents,
            embedder=stub_embedder,
            batch_embedder=batch_embedder or stub_batch_embedder,
            warm_up_fn=Mock(),
            now_fn=lambda: "2026-09-01 12:00:00",
        )
        created.append(service)

        return service

    yield build, legacy_log, dataset, incidents_path, cache_path

    for service in created:
        service.close()


def write_legacy_cache(cache_path, messages):
    """Write a cache in the OLD format, tagged with the old source."""
    save_embeddings(
        [
            EmbeddedSentence(
                sentence=message,
                source_text="satellite_faults.log",
                offset=index,
                embedding=letter_vector(message),
            )
            for index, message in enumerate(messages, start=1)
        ],
        cache_path,
    )


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def test_a_legacy_log_without_an_incident_history_is_migrated(upgrade):
    build, _, _, incidents_path, _ = upgrade
    service = build()

    service.refresh()

    assert service.migrated_from_legacy is True
    assert incidents_path.is_file()
    assert len(service) == 6


def test_migration_preserves_the_faults_learned_at_runtime(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    messages = [one.message for one in service.incidents]

    assert "Solar array deployment stalled midway" in messages
    assert "Star tracker lost its attitude reference" in messages
    assert "Downlink margin degraded during the pass" in messages


def test_the_prepared_dataset_is_not_imported_a_second_time(upgrade):
    """The legacy log already contains it; importing again would double."""
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    messages = [one.message for one in service.incidents]

    assert messages.count("Battery voltage dropped below safe level") == 1
    assert messages.count("Communication link lost") == 1
    assert len(service) == 6


def test_migration_produces_structured_incidents(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    first = service.incidents[0]

    assert first.incident_id == 1
    assert first.message == "Battery voltage dropped below safe level"
    assert first.subsystem == "UNKNOWN"
    assert first.error_code is None
    assert first.previous_action is None
    assert first.outcome is None
    assert first.count == 1


def test_migration_preserves_timestamps(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    by_message = {one.message: one for one in service.incidents}
    learned = by_message["Star tracker lost its attitude reference"]

    assert learned.first_seen == "2026-08-02 11:30:00"
    assert learned.last_seen == "2026-08-02 11:30:00"


def test_migration_preserves_severity(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    by_message = {one.message: one for one in service.incidents}

    assert by_message["Communication link lost"].severity == "CRITICAL"
    assert (
        by_message["Reaction wheel temperature too high"].severity == "WARNING"
    )
    assert (
        by_message["Solar array deployment stalled midway"].severity == "ERROR"
    )


def test_nothing_is_invented_during_migration(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    assert all(one.subsystem == "UNKNOWN" for one in service.incidents)
    assert all(one.error_code is None for one in service.incidents)
    assert all(one.previous_action is None for one in service.incidents)
    assert all(one.outcome is None for one in service.incidents)


# ----------------------------------------------------------------------
# Folding exact repeats
# ----------------------------------------------------------------------


def test_exact_repeats_become_one_incident_with_a_count(upgrade):
    build, legacy_log, _, _, _ = upgrade
    legacy_log.write_text(
        "2026-01-01 00:00:00 | ERROR | Transponder lock lost\n"
        "2026-05-05 12:00:00 | ERROR | Transponder lock lost\n"
        "2026-09-09 23:59:59 | ERROR | Transponder lock lost\n",
        encoding="utf-8",
    )
    service = build()
    service.refresh()

    assert len(service) == 1
    only = service.incidents[0]
    assert only.count == 3
    assert only.first_seen == "2026-01-01 00:00:00"
    assert only.last_seen == "2026-09-09 23:59:59"


def test_the_same_text_at_different_severities_stays_separate(upgrade):
    """Conservative: a WARNING and an ERROR are not the same incident."""
    build, legacy_log, _, _, _ = upgrade
    legacy_log.write_text(
        "2026-01-01 00:00:00 | WARNING | Battery voltage low\n"
        "2026-05-05 12:00:00 | ERROR | Battery voltage low\n",
        encoding="utf-8",
    )
    service = build()
    service.refresh()

    assert len(service) == 2


def test_similar_but_different_wording_is_never_merged(upgrade):
    """Migration folds identical text only, never a similarity judgement."""
    build, legacy_log, _, _, _ = upgrade
    legacy_log.write_text(
        "2026-01-01 00:00:00 | ERROR | Transponder lock lost\n"
        "2026-05-05 12:00:00 | ERROR | Transponder lock lost again\n",
        encoding="utf-8",
    )
    service = build()
    service.refresh()

    assert len(service) == 2


# ----------------------------------------------------------------------
# The old cache
# ----------------------------------------------------------------------


def test_an_old_cache_does_not_block_startup(upgrade):
    build, _, _, _, cache_path = upgrade
    write_legacy_cache(cache_path, ["Battery voltage dropped below safe level"])

    # Constructing used to raise CacheError here.
    service = build()

    assert service._legacy_cache is True
    assert len(service) == 0


def test_an_old_cache_is_replaced_by_a_rebuilt_one(upgrade):
    build, _, _, _, cache_path = upgrade
    write_legacy_cache(cache_path, ["Battery voltage dropped below safe level"])
    service = build()

    embedded = service.refresh()

    assert embedded == 6
    assert len(service._store) == 6
    assert service._store.metadata(0)["source_text"] == "satellite_incidents"
    assert not Path(f"{cache_path}.legacy").exists()


def test_a_cache_from_a_genuinely_foreign_source_is_still_refused(upgrade):
    """The exemption covers this project's own old layout, nothing else."""
    build, _, _, _, cache_path = upgrade
    save_embeddings(
        [
            EmbeddedSentence(
                sentence="Something else entirely",
                source_text="somebody_elses_corpus.txt",
                offset=1,
                embedding=[1.0, 2.0],
            )
        ],
        cache_path,
    )

    with pytest.raises(CacheError, match="was built from"):
        build()


def test_the_vector_count_matches_the_incident_count(upgrade):
    build, _, _, _, cache_path = upgrade
    write_legacy_cache(cache_path, ["Battery voltage dropped below safe level"])
    service = build()

    service.refresh()

    assert len(service._store) == len(service) == 6


# ----------------------------------------------------------------------
# Crash safety and evidence
# ----------------------------------------------------------------------


def test_the_legacy_log_is_never_deleted(upgrade):
    build, legacy_log, _, _, _ = upgrade
    service = build()
    service.refresh()

    assert legacy_log.is_file()
    assert legacy_log.read_text(encoding="utf-8") == LEGACY_LOG


def test_canonical_history_is_written_before_the_cache_is_touched(upgrade):
    """A failed rebuild must still leave a complete, correct history."""
    build, legacy_log, _, incidents_path, cache_path = upgrade
    write_legacy_cache(cache_path, ["Battery voltage dropped below safe level"])
    exploding = Mock(side_effect=RuntimeError("the model died"))
    service = build(batch_embedder=exploding)

    with pytest.raises(RuntimeError, match="the model died"):
        service.refresh()

    # Canonical history survived the failure in full.
    assert incidents_path.is_file()
    assert len(IncidentHistory(str(incidents_path)).load()) == 6
    # And so did the evidence it was built from.
    assert legacy_log.read_text(encoding="utf-8") == LEGACY_LOG


def test_a_failed_migration_can_be_retried(upgrade):
    build, _, _, _, cache_path = upgrade
    write_legacy_cache(cache_path, ["Battery voltage dropped below safe level"])
    failing = build(batch_embedder=Mock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        failing.refresh()
    failing.close()

    retried = build()

    assert retried.refresh() == 6
    assert len(retried._store) == len(retried) == 6


# ----------------------------------------------------------------------
# Idempotence
# ----------------------------------------------------------------------


def test_a_second_start_does_not_migrate_again(upgrade):
    build, _, _, _, _ = upgrade
    build().refresh()

    second = build()
    embedded = second.refresh()

    assert second.migrated_from_legacy is False
    assert embedded == 0
    assert len(second) == 6


def test_a_second_start_never_reads_the_legacy_log(upgrade, monkeypatch):
    build, legacy_log, _, _, _ = upgrade
    build().refresh()

    from src.logsearch import log_index

    read_paths = []
    original = log_index.parse_fault_records

    def spy(path, *arguments, **keywords):
        read_paths.append(str(path))
        return original(path, *arguments, **keywords)

    monkeypatch.setattr(log_index, "parse_fault_records", spy)
    second = build()
    second.refresh()

    assert read_paths == []


def test_faults_learned_after_migration_survive_a_restart(upgrade):
    build, _, _, _, _ = upgrade
    first = build()
    first.refresh()
    first.record_error("A fault discovered after the upgrade", severity="ERROR")
    first.close()

    restarted = build()

    assert restarted.refresh() == 0
    assert len(restarted) == 7
    assert any(
        one.message == "A fault discovered after the upgrade"
        for one in restarted.incidents
    )


def test_the_migration_is_stable_across_many_restarts(upgrade):
    build, _, _, _, _ = upgrade

    for _ in range(4):
        service = build()
        service.refresh()
        assert len(service) == 6
        service.close()


# ----------------------------------------------------------------------
# Bounded storage and ids
# ----------------------------------------------------------------------


def test_migration_respects_the_configured_capacity(upgrade):
    build, _, _, _, _ = upgrade
    service = build(max_incidents=4)

    service.refresh()

    assert len(service) <= 4
    assert len(service._store) == len(service)


def test_a_bounded_migration_keeps_the_critical_faults(upgrade):
    build, _, _, _, _ = upgrade
    service = build(max_incidents=4)

    service.refresh()

    severities = [one.severity for one in service.incidents]
    assert severities.count("CRITICAL") == 2


def test_migrated_ids_are_monotonic_and_continue_afterwards(upgrade):
    build, _, _, _, _ = upgrade
    service = build()
    service.refresh()

    ids = [one.incident_id for one in service.incidents]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)

    outcome = service.record_error("A new post-upgrade fault", severity="ERROR")
    assert outcome.incident_id > max(ids)


def test_the_id_counter_is_written_during_migration(upgrade):
    build, _, _, incidents_path, _ = upgrade
    build().refresh()

    assert IncidentHistory(str(incidents_path)).meta_path.is_file()


def test_no_legacy_log_means_the_dataset_is_used(upgrade, tmp_path):
    """Without a legacy history, a fresh install still bootstraps."""
    build, _, _, _, _ = upgrade
    service = build(legacy_log_path=tmp_path / "absent.log")

    service.refresh()

    assert service.migrated_from_legacy is False
    assert len(service) == 3
