"""Tests for the structured incident model and deduplication rules."""

import pytest

from src.logsearch.incidents import (
    DEDUP_SIMILARITY_THRESHOLD,
    FaultIncident,
    IncidentError,
    IncidentMatch,
    SEVERITIES,
    SUBSYSTEMS,
    find_duplicate,
    normalize_subsystem,
)


def incident(
    incident_id=1,
    message="Communication link lost",
    subsystem="COMMUNICATION",
    severity="CRITICAL",
    error_code=None,
    count=1,
):
    return FaultIncident(
        incident_id=incident_id,
        message=message,
        subsystem=subsystem,
        severity=severity,
        error_code=error_code,
        first_seen="2026-05-01 00:00:00",
        last_seen="2026-08-20 00:00:00",
        count=count,
    )


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------


def test_a_new_incident_starts_at_one_occurrence():
    new = FaultIncident(
        incident_id=1,
        message="Battery voltage low",
        first_seen="2026-09-01 10:00:00",
        last_seen="2026-09-01 10:00:00",
    )

    assert new.count == 1
    assert new.first_seen == new.last_seen


def test_the_defaults_are_unknown_subsystem_and_error_severity():
    new = FaultIncident(incident_id=1, message="Something failed")

    assert new.subsystem == "UNKNOWN"
    assert new.severity == "ERROR"
    assert new.error_code is None
    assert new.previous_action is None
    assert new.outcome is None


def test_every_required_subsystem_is_supported():
    assert set(SUBSYSTEMS) >= {
        "COMMUNICATION",
        "POWER",
        "THERMAL",
        "ATTITUDE_CONTROL",
        "PAYLOAD",
        "COMPUTE",
        "UNKNOWN",
    }


def test_every_required_severity_is_supported():
    assert set(SEVERITIES) == {"WARNING", "ERROR", "CRITICAL"}


def test_an_unknown_subsystem_is_rejected():
    with pytest.raises(IncidentError, match="not a known subsystem"):
        FaultIncident(incident_id=1, message="x", subsystem="PROPULSION")


def test_an_unknown_severity_is_rejected():
    with pytest.raises(IncidentError, match="not a known severity"):
        FaultIncident(incident_id=1, message="x", severity="FATAL")


def test_a_blank_message_is_rejected():
    with pytest.raises(IncidentError, match="must have a message"):
        FaultIncident(incident_id=1, message="   ")


def test_a_missing_subsystem_becomes_unknown():
    assert normalize_subsystem(None) == "UNKNOWN"
    assert normalize_subsystem("") == "UNKNOWN"
    assert normalize_subsystem("   ") == "UNKNOWN"


def test_a_subsystem_is_accepted_in_any_case():
    assert normalize_subsystem("power") == "POWER"
    assert normalize_subsystem(" Communication ") == "COMMUNICATION"


def test_a_recurrence_bumps_the_count_and_last_seen_only():
    existing = incident(count=12)

    existing.record_recurrence("2026-09-01 12:00:00")

    assert existing.count == 13
    assert existing.last_seen == "2026-09-01 12:00:00"
    assert existing.first_seen == "2026-05-01 00:00:00"


def test_an_incident_survives_a_round_trip():
    original = incident(error_code="COM-LINK-LOST", count=7)
    original.previous_action = "Switched to backup transponder"
    original.outcome = "Link restored"

    assert FaultIncident.from_dict(original.to_dict()) == original


def test_an_incident_written_by_an_older_version_still_loads():
    restored = FaultIncident.from_dict(
        {"incident_id": 3, "message": "Old style record"}
    )

    assert restored.subsystem == "UNKNOWN"
    assert restored.severity == "ERROR"
    assert restored.count == 1


def test_a_stored_incident_without_a_message_is_rejected():
    with pytest.raises(IncidentError, match="missing"):
        FaultIncident.from_dict({"incident_id": 3})


def test_a_match_exposes_the_incident_metadata():
    original = incident(error_code="COM-LINK-LOST", count=13)
    original.previous_action = "Power cycled the radio"
    original.outcome = "Recovered"

    match = IncidentMatch(incident=original, similarity=0.88)

    assert match.message == "Communication link lost"
    assert match.similarity == 0.88
    assert match.subsystem == "COMMUNICATION"
    assert match.severity == "CRITICAL"
    assert match.count == 13
    assert match.error_code == "COM-LINK-LOST"
    assert match.first_seen == "2026-05-01 00:00:00"
    assert match.last_seen == "2026-08-20 00:00:00"
    assert match.previous_action == "Power cycled the radio"
    assert match.outcome == "Recovered"


# ----------------------------------------------------------------------
# Deduplication: Case A, the error code
# ----------------------------------------------------------------------


def test_the_same_error_code_and_subsystem_deduplicates():
    existing = incident(error_code="PWR-LOW-VOLTAGE", subsystem="POWER")
    candidate = incident(
        incident_id=2,
        message="Completely different wording",
        subsystem="POWER",
        error_code="PWR-LOW-VOLTAGE",
    )

    # No similarity at all: the error code is an explicit identity claim.
    assert find_duplicate(candidate, [existing], {}) is existing


def test_the_same_error_code_with_a_different_subsystem_does_not_merge():
    existing = incident(error_code="PWR-LOW-VOLTAGE", subsystem="POWER")
    candidate = incident(
        incident_id=2,
        subsystem="THERMAL",
        error_code="PWR-LOW-VOLTAGE",
    )

    assert find_duplicate(candidate, [existing], {1: 0.99}) is None


def test_different_error_codes_do_not_merge():
    existing = incident(error_code="PWR-LOW-VOLTAGE", subsystem="POWER")
    candidate = incident(
        incident_id=2,
        subsystem="POWER",
        error_code="PWR-CELL-IMBALANCE",
    )

    # Even at near-identical wording, two explicit codes disagree.
    assert find_duplicate(candidate, [existing], {1: 0.99}) is None


# ----------------------------------------------------------------------
# Deduplication: Case B, semantic similarity
# ----------------------------------------------------------------------


def test_high_similarity_in_the_same_subsystem_deduplicates():
    existing = incident(subsystem="COMMUNICATION")
    candidate = incident(
        incident_id=2,
        message="Ground communication link lost",
        subsystem="COMMUNICATION",
    )

    assert find_duplicate(candidate, [existing], {1: 0.94}) is existing


def test_similarity_exactly_on_the_dedup_threshold_deduplicates():
    existing = incident(subsystem="COMMUNICATION")
    candidate = incident(incident_id=2, subsystem="COMMUNICATION")

    assert (
        find_duplicate(candidate, [existing], {1: DEDUP_SIMILARITY_THRESHOLD})
        is existing
    )


def test_similarity_below_the_dedup_threshold_creates_a_new_incident():
    existing = incident(subsystem="COMMUNICATION")
    candidate = incident(incident_id=2, subsystem="COMMUNICATION")

    assert find_duplicate(candidate, [existing], {1: 0.899}) is None


def test_the_search_threshold_does_not_imply_deduplication():
    """0.35 is good enough to display, nowhere near enough to merge."""
    existing = incident(subsystem="POWER")
    candidate = incident(incident_id=2, subsystem="POWER")

    for similarity in (0.35, 0.5, 0.72, 0.89):
        assert find_duplicate(candidate, [existing], {1: similarity}) is None


def test_high_similarity_across_subsystems_does_not_merge():
    existing = incident(subsystem="POWER")
    candidate = incident(incident_id=2, subsystem="THERMAL")

    assert find_duplicate(candidate, [existing], {1: 0.99}) is None


def test_a_conflicting_error_code_blocks_a_similarity_merge():
    existing = incident(subsystem="POWER", error_code="PWR-A")
    candidate = incident(incident_id=2, subsystem="POWER", error_code="PWR-B")

    assert find_duplicate(candidate, [existing], {1: 0.99}) is None


def test_a_missing_error_code_does_not_block_a_similarity_merge():
    existing = incident(subsystem="POWER", error_code=None)
    candidate = incident(incident_id=2, subsystem="POWER", error_code="PWR-A")

    assert find_duplicate(candidate, [existing], {1: 0.95}) is existing


def test_the_strongest_qualifying_candidate_wins():
    weaker = incident(incident_id=1, subsystem="POWER")
    stronger = incident(incident_id=2, subsystem="POWER")
    candidate = incident(incident_id=3, subsystem="POWER")

    duplicate = find_duplicate(
        candidate,
        [weaker, stronger],
        {1: 0.91, 2: 0.97},
    )

    assert duplicate is stronger


def test_an_unscored_incident_is_never_merged():
    """An incident the search never ranked cannot be a Case B duplicate."""
    existing = incident(subsystem="POWER")
    candidate = incident(incident_id=2, subsystem="POWER")

    assert find_duplicate(candidate, [existing], {}) is None
