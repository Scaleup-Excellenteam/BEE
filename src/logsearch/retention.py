"""Bounded storage policy for the semantic fault memory.

Fault memory cannot grow forever on a spacecraft, so the history has a
configured capacity and a deterministic rule for what to drop when that
capacity is reached.

Retention is NOT ranking
------------------------

This module decides what knowledge is worth KEEPING when storage runs
out.  It has nothing to do with which incidents a search returns: search
ranks by semantic similarity and only by semantic similarity.  ``count``
and ``last_seen`` appear here and nowhere near the ranking path.

Retention value, least valuable first
-------------------------------------

    (severity, count, last_seen, first_seen, incident_id)

Read left to right, that says: drop a WARNING before an ERROR and an
ERROR before a CRITICAL; within one severity drop the one that has
happened least; then the one not seen for longest; then the one that
started longest ago; and if two are still indistinguishable, the lower
id, purely so the outcome is reproducible.

The critical reserve
--------------------

A slice of capacity is held back so that routine WARNING noise cannot
crowd out the faults that actually endanger the mission.  Non-critical
incidents may never occupy those slots; CRITICAL incidents may use the
whole capacity.

On a real spacecraft the capacity and this policy would be mission
configured, and quite possibly different per subsystem.  The numbers
here are defaults, not physics.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.logsearch.incidents import FaultIncident


# How many incidents the memory may hold at once.
DEFAULT_MAX_INCIDENTS = 100_000

# The share of capacity reserved for CRITICAL incidents.
CRITICAL_RESERVE_RATIO = 0.10

# Ordering used when deciding what to drop.  Lower is dropped sooner.
SEVERITY_RANK = {"WARNING": 0, "ERROR": 1, "CRITICAL": 2}

CRITICAL = "CRITICAL"

# What happened to storage when a fault was reported.
NEW_INCIDENT_STORED = "NEW_INCIDENT_STORED"
DUPLICATE_UPDATED = "DUPLICATE_UPDATED"
NOT_STORED_CAPACITY = "NOT_STORED_CAPACITY"
EVICTED_AND_STORED = "EVICTED_AND_STORED"


@dataclass(frozen=True)
class AdmissionPlan:
    """What to do with an incoming incident at the current occupancy."""

    store: bool
    evict: FaultIncident | None
    reason: str


def retention_key(incident: FaultIncident) -> tuple:
    """Return the sort key that orders incidents least valuable first."""
    return (
        SEVERITY_RANK[incident.severity],
        incident.count,
        incident.last_seen,
        incident.first_seen,
        incident.incident_id,
    )


def critical_reserved_slots(
    max_incidents: int,
    ratio: float = CRITICAL_RESERVE_RATIO,
) -> int:
    """Return how many slots only CRITICAL incidents may occupy.

    Always at least one slot whenever there is any capacity at all: a
    reserve that rounds down to zero would not be a reserve.
    """
    if max_incidents <= 0:
        return 0

    return max(1, int(max_incidents * ratio))


def noncritical_capacity(
    max_incidents: int,
    ratio: float = CRITICAL_RESERVE_RATIO,
) -> int:
    """Return how many slots WARNING and ERROR incidents may occupy."""
    return max_incidents - critical_reserved_slots(max_incidents, ratio)


def _count_noncritical(incidents: list[FaultIncident]) -> int:
    return sum(1 for one in incidents if one.severity != CRITICAL)


def least_valuable(
    incidents: list[FaultIncident],
    non_critical_only: bool = False,
) -> FaultIncident | None:
    """Return the incident that should be dropped first, if any."""
    eligible = [
        one
        for one in incidents
        if not non_critical_only or one.severity != CRITICAL
    ]

    if not eligible:
        return None

    return min(eligible, key=retention_key)


def plan_admission(
    candidate: FaultIncident,
    incidents: list[FaultIncident],
    max_incidents: int,
    ratio: float = CRITICAL_RESERVE_RATIO,
) -> AdmissionPlan:
    """Decide whether a genuinely new incident can be stored.

    A CRITICAL fault is always stored: if memory is full it displaces the
    least valuable incident anywhere in the history, which is a
    lower-severity one whenever such an incident exists.

    A WARNING or ERROR is stored only while non-critical capacity
    remains.  Once it is exhausted the newcomer is weighed against the
    weakest non-critical incident already held, using the same retention
    key.  It is admitted only if that incident is strictly less valuable
    than itself; otherwise the knowledge already in memory is worth more
    than the newcomer and the newcomer is not persisted.  CRITICAL
    incidents are never evicted to make room for a non-critical one.
    """
    if max_incidents <= 0:
        return AdmissionPlan(False, None, NOT_STORED_CAPACITY)

    occupied = len(incidents)

    if candidate.severity == CRITICAL:
        if occupied < max_incidents:
            return AdmissionPlan(True, None, NEW_INCIDENT_STORED)

        victim = least_valuable(incidents)

        if victim is None:
            return AdmissionPlan(False, None, NOT_STORED_CAPACITY)

        return AdmissionPlan(True, victim, EVICTED_AND_STORED)

    room_overall = occupied < max_incidents
    room_for_noncritical = _count_noncritical(incidents) < noncritical_capacity(
        max_incidents,
        ratio,
    )

    if room_overall and room_for_noncritical:
        return AdmissionPlan(True, None, NEW_INCIDENT_STORED)

    victim = least_valuable(incidents, non_critical_only=True)

    if victim is None:
        # Everything held is CRITICAL, so there is nothing this fault is
        # allowed to displace.
        return AdmissionPlan(False, None, NOT_STORED_CAPACITY)

    if retention_key(victim) < retention_key(candidate):
        return AdmissionPlan(True, victim, EVICTED_AND_STORED)

    return AdmissionPlan(False, None, NOT_STORED_CAPACITY)


def select_within_capacity(
    incidents: list[FaultIncident],
    max_incidents: int,
    ratio: float = CRITICAL_RESERVE_RATIO,
) -> list[FaultIncident]:
    """Return the most valuable incidents that fit inside the budget.

    Used when a prepared dataset is larger than the configured capacity.
    The same two bounds apply as at runtime: the total, and the share
    non-critical incidents may occupy.  The survivors are returned in
    incident id order so the history file stays ordered.
    """
    if max_incidents <= 0:
        return []

    allowed_noncritical = noncritical_capacity(max_incidents, ratio)
    kept: list[FaultIncident] = []
    noncritical_kept = 0

    # Most valuable first, so the budget is spent on what matters most.
    for incident in sorted(incidents, key=retention_key, reverse=True):
        if len(kept) >= max_incidents:
            break

        if incident.severity != CRITICAL:
            if noncritical_kept >= allowed_noncritical:
                continue

            noncritical_kept += 1

        kept.append(incident)

    return sorted(kept, key=lambda one: one.incident_id)
