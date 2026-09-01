"""Tests for bounded fault memory at the service level.

Capacities here are tiny (3, 5, 10) so the boundary behaviour is visible
and deterministic.  Nothing depends on the production default.

Most tests set ``dedup_threshold`` above 1.0 so that every reported fault
counts as new: this keeps the capacity rules under test rather than the
deduplication rules, which have their own tests.
"""

import json
from collections import Counter
from unittest.mock import Mock

import pytest

from src.logsearch.incident_history import IncidentHistory
from src.logsearch.incidents import FaultIncident
from src.logsearch.log_index import LogSearchService
from src.logsearch.retention import (
    EVICTED_AND_STORED,
    NEW_INCIDENT_STORED,
    NOT_STORED_CAPACITY,
)


NEVER_DEDUP = 1.01


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
def memory(tmp_path):
    """Build services sharing one history and cache, with no dataset."""
    incidents_path = tmp_path / "logs" / "satellite_incidents.jsonl"
    cache_path = str(tmp_path / "satellite_fault_cache")
    created = []
    tick = [0]

    def clock():
        tick[0] += 1
        return f"2026-09-01 10:00:{tick[0]:02d}"

    def build(
        max_incidents=5,
        dedup_threshold=NEVER_DEDUP,
        batch_embedder=None,
        dataset=None,
    ):
        service = LogSearchService(
            incidents_path=str(incidents_path),
            cache_path=cache_path,
            dataset_path=str(dataset) if dataset else None,
            legacy_log_path=None,
            max_incidents=max_incidents,
            dedup_threshold=dedup_threshold,
            embedder=stub_embedder,
            batch_embedder=batch_embedder or stub_batch_embedder,
            warm_up_fn=Mock(),
            now_fn=clock,
        )
        created.append(service)

        return service

    yield build, incidents_path, cache_path

    for service in created:
        service.close()


def seed(incidents_path, incidents, next_incident_id=None):
    """Write a history file directly, to set up an exact starting state."""
    history = IncidentHistory(str(incidents_path))
    history.save(
        incidents,
        next_incident_id or (max(one.incident_id for one in incidents) + 1),
    )


def incident(
    incident_id,
    message=None,
    severity="WARNING",
    count=1,
    last_seen="2026-01-01 00:00:00",
):
    return FaultIncident(
        incident_id=incident_id,
        message=message or f"Historical fault number {incident_id}",
        severity=severity,
        count=count,
        first_seen="2026-01-01 00:00:00",
        last_seen=last_seen,
    )


def fill(service, count, severity="WARNING", prefix="Fault"):
    """Report ``count`` distinct faults and return their outcomes."""
    return [
        service.record_error(f"{prefix} number {index}", severity=severity)
        for index in range(count)
    ]


# ----------------------------------------------------------------------
# The bound itself
# ----------------------------------------------------------------------


def test_capacity_is_never_exceeded(memory):
    build, incidents_path, _ = memory
    service = build(max_incidents=5)
    service.refresh()

    for index in range(40):
        service.record_error(
            f"Fault number {index}",
            severity=["WARNING", "ERROR", "CRITICAL"][index % 3],
        )
        assert len(service) <= 5

    assert len(service) == 5
    stored = incidents_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(stored) == 5
    assert len(service._store) == 5


def test_the_jsonl_and_the_cache_never_disagree(memory):
    build, _, _ = memory
    service = build(max_incidents=3)
    service.refresh()

    for index in range(15):
        service.record_error(f"Fault number {index}", severity="ERROR")
        assert len(service._store) == len(service)


# Messages with genuinely different letter profiles, so the stub only
# calls something a duplicate when it is repeated word for word.
DISTINCT_FAULTS = [
    "Battery undervoltage on the main bus",
    "Gyroscope spin rate exceeded its limit",
    "Payload heater valve jammed wide open",
]


def test_a_duplicate_at_full_capacity_still_updates_the_count(memory):
    build, _, _ = memory
    service = build(max_incidents=3, dedup_threshold=0.999)
    service.refresh()
    for message in DISTINCT_FAULTS:
        service.record_error(message, severity="CRITICAL")
    assert len(service) == 3

    outcome = service.record_error(DISTINCT_FAULTS[0], severity="CRITICAL")

    assert outcome.deduplicated is True
    assert outcome.stored is True
    assert len(service) == 3
    assert any(one.count == 2 for one in service.incidents)


def test_a_duplicate_at_full_capacity_creates_no_embedding(memory):
    build, _, _ = memory
    service = build(max_incidents=3, dedup_threshold=0.999)
    service.refresh()
    for message in DISTINCT_FAULTS:
        service.record_error(message, severity="CRITICAL")
    vectors_before = len(service._store)

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service.batch_embedder = batch_embedder
    service.record_error(DISTINCT_FAULTS[0], severity="CRITICAL")

    batch_embedder.assert_not_called()
    assert len(service._store) == vectors_before == 3


# ----------------------------------------------------------------------
# The critical reserve
# ----------------------------------------------------------------------


def test_noncritical_faults_cannot_consume_the_reserve(memory):
    build, _, _ = memory
    service = build(max_incidents=10)
    service.refresh()

    fill(service, 20, severity="WARNING")

    # Nine slots at most, because one is reserved for CRITICAL.
    assert len(service) == 9
    assert service.storage_status()["critical_reserved_slots"] == 1


def test_a_critical_can_still_be_stored_when_noncriticals_are_full(memory):
    build, _, _ = memory
    service = build(max_incidents=10)
    service.refresh()
    fill(service, 20, severity="WARNING")

    outcome = service.record_error("Reactor breach", severity="CRITICAL")

    assert outcome.stored is True
    assert outcome.evicted_incident_id is None
    assert outcome.reason == NEW_INCIDENT_STORED
    assert len(service) == 10


def test_a_warning_is_evicted_before_an_error(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, severity="WARNING", count=1),
            incident(2, severity="ERROR", count=1),
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("A brand new fault", severity="ERROR")

    assert outcome.evicted_incident_id == 1
    assert [one.severity for one in service.incidents] == ["ERROR", "ERROR"]


def test_an_error_is_evicted_before_a_critical(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, severity="ERROR", count=1),
            incident(2, severity="CRITICAL", count=1),
        ],
    )
    service = build(max_incidents=2)
    service.refresh()

    outcome = service.record_error("A brand new fault", severity="CRITICAL")

    assert outcome.evicted_incident_id == 1
    assert [one.severity for one in service.incidents] == [
        "CRITICAL",
        "CRITICAL",
    ]


def test_within_one_severity_the_rarest_is_evicted(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, severity="ERROR", count=9),
            incident(2, severity="ERROR", count=1),
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("A brand new fault", severity="ERROR")

    assert outcome.evicted_incident_id == 2


def test_at_equal_count_the_stalest_is_evicted(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(
                1, severity="ERROR", count=1, last_seen="2026-08-01 00:00:00"
            ),
            incident(
                2, severity="ERROR", count=1, last_seen="2026-01-01 00:00:00"
            ),
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("A brand new fault", severity="ERROR")

    assert outcome.evicted_incident_id == 2


def test_eviction_is_deterministic_across_identical_runs(memory, tmp_path):
    """Two identical histories must evict exactly the same incident."""
    evicted = []

    for run in range(2):
        incidents_path = tmp_path / f"run{run}" / "incidents.jsonl"
        seed(
            incidents_path,
            [incident(index, severity="ERROR", count=1) for index in (3, 5, 9)],
        )
        service = LogSearchService(
            incidents_path=str(incidents_path),
            cache_path=str(tmp_path / f"run{run}" / "cache"),
            dataset_path=None,
            legacy_log_path=None,
            max_incidents=4,
            dedup_threshold=NEVER_DEDUP,
            embedder=stub_embedder,
            batch_embedder=stub_batch_embedder,
            warm_up_fn=Mock(),
            now_fn=lambda: "2026-09-01 10:00:00",
        )
        service.refresh()
        evicted.append(
            service.record_error("A new fault", severity="ERROR")
            .evicted_incident_id
        )
        service.close()

    assert evicted[0] == evicted[1] == 3


# ----------------------------------------------------------------------
# Memory full of CRITICAL incidents
# ----------------------------------------------------------------------


def test_a_warning_is_refused_when_every_slot_is_critical(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [incident(index, severity="CRITICAL") for index in range(1, 4)],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("A routine warning", severity="WARNING")

    assert outcome.stored is False
    assert outcome.deduplicated is False
    assert outcome.incident_id is None
    assert outcome.evicted_incident_id is None
    assert outcome.reason == NOT_STORED_CAPACITY
    assert len(service) == 3


def test_a_refused_fault_still_returns_its_search_results(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(index, message=f"Critical fault {index}", severity="CRITICAL")
            for index in range(1, 4)
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("Critical fault 1", severity="WARNING")

    assert outcome.stored is False
    # The operator still gets their answer.
    assert outcome.matches
    assert outcome.matches[0].message == "Critical fault 1"


def test_a_critical_evicts_a_critical_when_memory_is_all_critical(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(index, severity="CRITICAL", count=index)
            for index in range(1, 4)
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    outcome = service.record_error("A brand new emergency", severity="CRITICAL")

    assert outcome.stored is True
    assert outcome.reason == EVICTED_AND_STORED
    assert outcome.evicted_incident_id == 1
    assert len(service) == 3
    assert len(service._store) == 3


def test_a_full_critical_memory_never_crashes(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [incident(index, severity="CRITICAL") for index in range(1, 4)],
    )
    service = build(max_incidents=3)
    service.refresh()

    for index in range(10):
        service.record_error(f"Warning number {index}", severity="WARNING")
        service.record_error(f"Emergency number {index}", severity="CRITICAL")

    assert len(service) == 3
    assert len(service._store) == 3


# ----------------------------------------------------------------------
# Search is unaffected by the storage decision
# ----------------------------------------------------------------------


def test_the_search_happens_before_the_storage_decision(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [incident(1, message="Battery voltage dropped", severity="CRITICAL")],
    )
    service = build(max_incidents=1)
    service.refresh()

    outcome = service.record_error("Battery voltage dropped", severity="WARNING")

    # Memory is full of criticals, so nothing was stored, yet the fault
    # was still matched against what history holds.
    assert outcome.stored is False
    assert [match.message for match in outcome.matches] == [
        "Battery voltage dropped"
    ]


def test_an_evicted_incident_disappears_from_later_searches(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, message="Doomed fault about gyroscopes", count=1),
            incident(2, message="Kept fault about transponders", count=9),
        ],
    )
    service = build(max_incidents=3)
    service.refresh()
    service.min_similarity = 0.0

    service.record_error("A brand new fault", severity="WARNING")

    found = {
        match.message for match in service.search_similar_logs("x", k=50)
    }
    assert "Doomed fault about gyroscopes" not in found
    assert "Kept fault about transponders" in found


def test_retained_incidents_are_still_returned_correctly(memory):
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, message="Doomed fault", count=1),
            incident(2, message="Kept fault about transponders", count=9),
        ],
    )
    service = build(max_incidents=3)
    service.refresh()

    service.record_error("A brand new fault", severity="WARNING")
    service.min_similarity = 0.0
    results = service.search_similar_logs("Kept fault about transponders", k=5)

    kept = [
        match
        for match in results
        if match.message == "Kept fault about transponders"
    ]
    assert kept, "the retained incident must still be searchable"
    assert kept[0].count == 9


def test_ranking_stays_semantic_under_a_bounded_memory(memory):
    """A frequent weak match must not outrank a strong one."""
    build, incidents_path, _ = memory
    seed(
        incidents_path,
        [
            incident(1, message="Reaction wheel bearing seized", count=900),
            incident(2, message="Transponder carrier lock lost", count=1),
        ],
    )
    service = build(max_incidents=10)
    service.refresh()

    results = service.search_similar_logs("Transponder carrier lock lost", k=5)

    assert results[0].message == "Transponder carrier lock lost"
    assert results[0].count == 1


# ----------------------------------------------------------------------
# Incident ids
# ----------------------------------------------------------------------


def test_ids_are_never_reused_after_an_eviction(memory):
    build, _, _ = memory
    service = build(max_incidents=3)
    service.refresh()

    issued = []
    for index in range(12):
        outcome = service.record_error(
            f"Fault number {index}",
            severity="CRITICAL",
        )
        if outcome.incident_id is not None:
            issued.append(outcome.incident_id)

    assert issued == sorted(issued)
    assert len(set(issued)) == len(issued)
    # Ids kept climbing even though only three incidents are ever held.
    assert max(issued) >= 12


def test_ids_survive_a_restart(memory):
    build, _, _ = memory
    first = build(max_incidents=3)
    first.refresh()
    fill(first, 5, severity="CRITICAL")
    highest = max(one.incident_id for one in first.incidents)
    first.close()

    restarted = build(max_incidents=3)
    outcome = restarted.record_error("Another fault", severity="CRITICAL")

    assert outcome.incident_id > highest


def test_evicting_the_newest_incident_does_not_free_its_id(memory, tmp_path):
    """The id counter, not the surviving ids, decides what comes next."""
    incidents_path = tmp_path / "incidents.jsonl"
    seed(incidents_path, [incident(7, severity="CRITICAL", count=1)], 8)

    service = LogSearchService(
        incidents_path=str(incidents_path),
        cache_path=str(tmp_path / "cache"),
        dataset_path=None,
        legacy_log_path=None,
        max_incidents=1,
        dedup_threshold=NEVER_DEDUP,
        embedder=stub_embedder,
        batch_embedder=stub_batch_embedder,
        warm_up_fn=Mock(),
        now_fn=lambda: "2026-09-01 10:00:00",
    )
    service.refresh()

    outcome = service.record_error("Replacement fault", severity="CRITICAL")

    assert outcome.evicted_incident_id == 7
    assert outcome.incident_id == 8
    service.close()


def test_the_id_counter_is_persisted_beside_the_history(memory):
    build, incidents_path, _ = memory
    service = build(max_incidents=3)
    service.refresh()
    fill(service, 4, severity="CRITICAL")

    meta = json.loads(
        IncidentHistory(str(incidents_path)).meta_path.read_text(
            encoding="utf-8"
        )
    )

    assert meta["next_incident_id"] > len(service)


# ----------------------------------------------------------------------
# Cache consistency around eviction
# ----------------------------------------------------------------------


def test_compaction_keeps_one_vector_per_incident(memory):
    build, _, cache_path = memory
    service = build(max_incidents=4)
    service.refresh()

    fill(service, 12, severity="CRITICAL")

    assert len(service._store) == len(service) == 4
    assert service.storage_status()["incidents"] == 4


def test_compaction_leaves_no_staging_directories(memory, tmp_path):
    build, _, cache_path = memory
    service = build(max_incidents=3)
    service.refresh()
    fill(service, 8, severity="CRITICAL")

    from pathlib import Path

    assert not Path(f"{cache_path}.rebuild").exists()
    assert not Path(f"{cache_path}.old").exists()


def test_compaction_does_not_re_embed_the_survivors(memory):
    """Eviction copies existing vectors; only the newcomer is embedded."""
    build, _, _ = memory
    service = build(max_incidents=3)
    service.refresh()
    fill(service, 3, severity="CRITICAL")

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    service.batch_embedder = batch_embedder
    service.record_error("A brand new emergency", severity="CRITICAL")

    embedded = [
        text
        for call in batch_embedder.call_args_list
        for text in call.args[0]
    ]
    assert embedded == ["A brand new emergency"]


def test_the_cache_survives_a_restart_after_compaction(memory):
    build, _, _ = memory
    first = build(max_incidents=3)
    first.refresh()
    fill(first, 8, severity="CRITICAL")
    surviving = [one.incident_id for one in first.incidents]
    first.close()

    batch_embedder = Mock(side_effect=stub_batch_embedder)
    restarted = build(max_incidents=3, batch_embedder=batch_embedder)

    assert restarted.refresh() == 0
    batch_embedder.assert_not_called()
    assert [one.incident_id for one in restarted.incidents] == surviving


def test_a_deleted_cache_is_rebuilt_after_compaction(memory):
    import shutil

    build, _, cache_path = memory
    first = build(max_incidents=3)
    first.refresh()
    fill(first, 8, severity="CRITICAL")
    first.close()

    shutil.rmtree(cache_path)
    rebuilt = build(max_incidents=3)

    assert rebuilt.refresh() == 3
    assert len(rebuilt._store) == len(rebuilt) == 3


# ----------------------------------------------------------------------
# Bootstrap and status
# ----------------------------------------------------------------------


def test_a_dataset_larger_than_capacity_is_bounded(memory, tmp_path):
    build, _, _ = memory
    dataset = tmp_path / "prepared.log"
    dataset.write_text(
        "".join(
            f"2024-03-{index:02d} 04:12:07 | ERROR | Historical fault {index}\n"
            for index in range(1, 16)
        ),
        encoding="utf-8",
    )

    service = build(max_incidents=5, dataset=dataset)
    embedded = service.refresh()

    assert len(service) == 4  # one slot reserved for CRITICAL
    assert embedded == 4
    assert len(service._store) == 4


def test_bootstrap_keeps_criticals_when_over_capacity(memory, tmp_path):
    build, _, _ = memory
    dataset = tmp_path / "prepared.log"
    dataset.write_text(
        "".join(
            f"2024-03-{index:02d} 04:12:07 | WARNING | Routine fault {index}\n"
            for index in range(1, 10)
        )
        + "2024-04-01 00:00:00 | CRITICAL | Attitude control lost\n",
        encoding="utf-8",
    )

    service = build(max_incidents=5, dataset=dataset)
    service.refresh()

    messages = [one.message for one in service.incidents]
    assert "Attitude control lost" in messages
    assert len(service) == 5


def test_storage_status_reports_the_budget(memory):
    build, _, _ = memory
    service = build(max_incidents=10)
    service.refresh()
    fill(service, 4, severity="WARNING")
    service.record_error("An emergency", severity="CRITICAL")

    assert service.storage_status() == {
        "incidents": 5,
        "max_incidents": 10,
        "usage_percent": 50.0,
        "critical_incidents": 1,
        "noncritical_incidents": 4,
        "critical_reserved_slots": 1,
    }


def test_storage_status_on_an_empty_memory(memory):
    build, _, _ = memory
    service = build(max_incidents=1000)
    service.refresh()

    status = service.storage_status()

    assert status["incidents"] == 0
    assert status["usage_percent"] == 0.0
    assert status["critical_reserved_slots"] == 100
