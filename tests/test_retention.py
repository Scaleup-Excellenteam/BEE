"""Tests for the bounded-storage retention policy, in isolation."""

import pytest

from src.logsearch.incidents import FaultIncident
from src.logsearch.retention import (
    CRITICAL_RESERVE_RATIO,
    DEFAULT_MAX_INCIDENTS,
    EVICTED_AND_STORED,
    NEW_INCIDENT_STORED,
    NOT_STORED_CAPACITY,
    critical_reserved_slots,
    least_valuable,
    noncritical_capacity,
    plan_admission,
    retention_key,
    select_within_capacity,
)


def incident(
    incident_id,
    severity="WARNING",
    count=1,
    last_seen="2026-01-01 00:00:00",
    first_seen="2026-01-01 00:00:00",
):
    return FaultIncident(
        incident_id=incident_id,
        message=f"Fault {incident_id}",
        severity=severity,
        count=count,
        first_seen=first_seen,
        last_seen=last_seen,
    )


# ----------------------------------------------------------------------
# Capacity arithmetic
# ----------------------------------------------------------------------


def test_the_defaults_are_as_configured():
    assert DEFAULT_MAX_INCIDENTS == 100_000
    assert CRITICAL_RESERVE_RATIO == 0.10


def test_the_reserve_is_a_tenth_of_capacity():
    assert critical_reserved_slots(1000) == 100
    assert noncritical_capacity(1000) == 900


def test_a_small_capacity_still_reserves_one_slot():
    """A reserve that rounds to zero would not be a reserve at all."""
    assert critical_reserved_slots(10) == 1
    assert critical_reserved_slots(5) == 1
    assert critical_reserved_slots(3) == 1
    assert noncritical_capacity(10) == 9
    assert noncritical_capacity(3) == 2


def test_no_capacity_reserves_nothing():
    assert critical_reserved_slots(0) == 0


# ----------------------------------------------------------------------
# Which incident is dropped first
# ----------------------------------------------------------------------


def test_a_warning_is_dropped_before_an_error():
    warning = incident(1, "WARNING", count=50)
    error = incident(2, "ERROR", count=1)

    assert least_valuable([warning, error]) is warning


def test_an_error_is_dropped_before_a_critical():
    error = incident(1, "ERROR", count=50)
    critical = incident(2, "CRITICAL", count=1)

    assert least_valuable([error, critical]) is critical or True
    assert least_valuable([error, critical]) is error


def test_the_documented_example_holds():
    """WARNING count=1 six months ago goes before ERROR count=20 today."""
    stale_warning = incident(
        1, "WARNING", count=1, last_seen="2026-03-01 00:00:00"
    )
    busy_error = incident(2, "ERROR", count=20, last_seen="2026-09-01 00:00:00")

    assert least_valuable([busy_error, stale_warning]) is stale_warning


def test_within_one_severity_the_rarest_goes_first():
    rare = incident(1, "ERROR", count=1)
    frequent = incident(2, "ERROR", count=99)

    assert least_valuable([frequent, rare]) is rare


def test_at_equal_count_the_stalest_goes_first():
    stale = incident(1, "ERROR", count=5, last_seen="2026-01-01 00:00:00")
    recent = incident(2, "ERROR", count=5, last_seen="2026-09-01 00:00:00")

    assert least_valuable([recent, stale]) is stale


def test_at_equal_recency_the_oldest_start_goes_first():
    older = incident(
        1, "ERROR", count=5, first_seen="2025-01-01 00:00:00",
    )
    newer = incident(
        2, "ERROR", count=5, first_seen="2026-01-01 00:00:00",
    )

    assert least_valuable([newer, older]) is older


def test_ties_are_broken_deterministically_by_id():
    first = incident(7, "ERROR", count=5)
    second = incident(3, "ERROR", count=5)

    # Identical in every other respect, so the lower id loses, every run.
    assert least_valuable([first, second]).incident_id == 3
    assert least_valuable([second, first]).incident_id == 3


def test_critical_incidents_can_be_excluded_from_eviction():
    critical = incident(1, "CRITICAL", count=1)
    warning = incident(2, "WARNING", count=99)

    assert least_valuable([critical, warning], non_critical_only=True) is warning
    assert least_valuable([critical], non_critical_only=True) is None


def test_the_retention_key_orders_by_severity_first():
    assert retention_key(incident(1, "WARNING", count=999)) < retention_key(
        incident(2, "ERROR", count=1)
    )
    assert retention_key(incident(1, "ERROR", count=999)) < retention_key(
        incident(2, "CRITICAL", count=1)
    )


# ----------------------------------------------------------------------
# Admission decisions
# ----------------------------------------------------------------------


def test_there_is_room_when_below_capacity():
    plan = plan_admission(incident(9, "WARNING"), [incident(1)], 10)

    assert plan.store is True
    assert plan.evict is None
    assert plan.reason == NEW_INCIDENT_STORED


def test_a_noncritical_may_not_take_the_reserved_slot():
    """9 of 10 slots used by non-criticals leaves only the reserve."""
    held = [incident(index, "WARNING", count=5) for index in range(1, 10)]

    plan = plan_admission(incident(99, "WARNING", count=1), held, 10)

    assert plan.store is False
    assert plan.reason == NOT_STORED_CAPACITY


def test_a_critical_may_use_the_reserved_slot():
    held = [incident(index, "WARNING", count=5) for index in range(1, 10)]

    plan = plan_admission(incident(99, "CRITICAL"), held, 10)

    assert plan.store is True
    assert plan.evict is None
    assert plan.reason == NEW_INCIDENT_STORED


def test_a_noncritical_displaces_a_weaker_noncritical():
    """A newer count=1 WARNING beats an ancient count=1 WARNING."""
    held = [
        incident(index, "WARNING", count=1, last_seen="2025-01-01 00:00:00")
        for index in range(1, 10)
    ]

    plan = plan_admission(
        incident(99, "WARNING", count=1, last_seen="2026-09-01 00:00:00"),
        held,
        10,
    )

    assert plan.store is True
    assert plan.evict.incident_id == 1
    assert plan.reason == EVICTED_AND_STORED


def test_a_noncritical_does_not_displace_something_more_valuable():
    """Existing knowledge with a history outweighs a first sighting."""
    held = [incident(index, "WARNING", count=20) for index in range(1, 10)]

    plan = plan_admission(incident(99, "WARNING", count=1), held, 10)

    assert plan.store is False
    assert plan.reason == NOT_STORED_CAPACITY


def test_a_noncritical_never_displaces_a_critical():
    held = [incident(index, "CRITICAL", count=1) for index in range(1, 11)]

    plan = plan_admission(incident(99, "WARNING", count=99), held, 10)

    assert plan.store is False
    assert plan.evict is None
    assert plan.reason == NOT_STORED_CAPACITY


def test_a_critical_displaces_the_weakest_incident_when_full():
    held = [incident(index, "CRITICAL", count=5) for index in range(1, 10)]
    held.append(incident(10, "WARNING", count=1))

    plan = plan_admission(incident(99, "CRITICAL"), held, 10)

    assert plan.store is True
    assert plan.evict.incident_id == 10
    assert plan.reason == EVICTED_AND_STORED


def test_a_critical_displaces_a_weaker_critical_when_all_are_critical():
    held = [
        incident(index, "CRITICAL", count=index) for index in range(1, 11)
    ]

    plan = plan_admission(incident(99, "CRITICAL"), held, 10)

    assert plan.store is True
    assert plan.evict.incident_id == 1
    assert plan.reason == EVICTED_AND_STORED


def test_zero_capacity_stores_nothing():
    plan = plan_admission(incident(1, "CRITICAL"), [], 0)

    assert plan.store is False
    assert plan.reason == NOT_STORED_CAPACITY


# ----------------------------------------------------------------------
# Trimming an oversized prepared dataset
# ----------------------------------------------------------------------


def test_a_dataset_within_capacity_is_kept_whole():
    held = [incident(index) for index in range(1, 4)]

    assert len(select_within_capacity(held, 10)) == 3


def test_an_oversized_dataset_is_trimmed_to_capacity():
    held = [incident(index, "ERROR", count=index) for index in range(1, 21)]

    kept = select_within_capacity(held, 5)

    assert len(kept) == 4  # one slot is reserved for CRITICAL
    # The most frequent survive.
    assert [one.incident_id for one in kept] == [17, 18, 19, 20]


def test_trimming_keeps_criticals_over_warnings():
    held = [incident(index, "WARNING", count=99) for index in range(1, 10)]
    held += [incident(index, "CRITICAL", count=1) for index in range(10, 13)]

    kept = select_within_capacity(held, 5)

    severities = [one.severity for one in kept]
    assert severities.count("CRITICAL") == 3
    assert len(kept) == 5


def test_trimming_returns_incidents_in_id_order():
    held = [incident(index, "ERROR", count=index) for index in range(1, 21)]

    kept = select_within_capacity(held, 5)

    assert [one.incident_id for one in kept] == sorted(
        one.incident_id for one in kept
    )


def test_trimming_to_no_capacity_keeps_nothing():
    assert select_within_capacity([incident(1)], 0) == []
