"""Structured satellite fault incidents.

A satellite does not have "log lines", it has INCIDENTS: a thing that
went wrong, in a subsystem, that may happen again.  The same fault
recurring two hundred times is one incident with a count of two hundred,
not two hundred records.

    FaultIncident   one thing that went wrong, plus how often
    IncidentMatch   an incident together with how similar it is to a query

Ranking note
------------

``count``, ``severity`` and ``last_seen`` are METADATA.  They are shown
to the operator and they are used to decide whether two reports are the
same incident, but they never influence search ranking: a weak match
must not float above a strong one just because it is frequent, severe or
recent.  Ranking is similarity, and only similarity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Subsystems a fault can be attributed to.  No classifier yet: the caller
# says which one, and anything unstated is UNKNOWN rather than guessed.
SUBSYSTEMS = (
    "COMMUNICATION",
    "POWER",
    "THERMAL",
    "ATTITUDE_CONTROL",
    "PAYLOAD",
    "COMPUTE",
    "UNKNOWN",
)
DEFAULT_SUBSYSTEM = "UNKNOWN"

# Severities, matching the log levels the parser already understands.
SEVERITIES = ("WARNING", "ERROR", "CRITICAL")
DEFAULT_SEVERITY = "ERROR"

# How alike two reports must be before they are treated as the SAME
# incident.  This is deliberately far above the 0.35 used for search:
# 0.35 answers "is this worth showing you?", which is a much weaker
# question than "is this literally the same fault?".  Merging two
# distinct faults destroys information that cannot be recovered, while
# an extra incident costs one embedding, so the bar is set high.
DEDUP_SIMILARITY_THRESHOLD = 0.90


class IncidentError(ValueError):
    """Raised when an incident is described with invalid values."""


@dataclass
class FaultIncident:
    """One distinct fault, and everything known about its recurrences."""

    incident_id: int
    message: str
    subsystem: str = DEFAULT_SUBSYSTEM
    severity: str = DEFAULT_SEVERITY
    error_code: str | None = None
    first_seen: str = ""
    last_seen: str = ""
    count: int = 1
    previous_action: str | None = None
    outcome: str | None = None
    source_text: str = ""
    source_offset: int = 0

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise IncidentError("An incident must have a message")

        if self.subsystem not in SUBSYSTEMS:
            raise IncidentError(
                f"{self.subsystem!r} is not a known subsystem; expected one "
                f"of {', '.join(SUBSYSTEMS)}"
            )

        if self.severity not in SEVERITIES:
            raise IncidentError(
                f"{self.severity!r} is not a known severity; expected one "
                f"of {', '.join(SEVERITIES)}"
            )

    def record_recurrence(self, timestamp: str) -> None:
        """Note that this incident has happened again.

        ``first_seen`` is never touched: when a fault started is a fact
        about history, and later occurrences do not change it.
        """
        self.count += 1
        self.last_seen = timestamp

    def to_dict(self) -> dict:
        """Return the incident as a plain JSON-ready dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> "FaultIncident":
        """Rebuild an incident from stored JSON."""
        known = {f: record.get(f) for f in cls.__dataclass_fields__}

        missing = [
            name
            for name in ("incident_id", "message")
            if known.get(name) is None
        ]
        if missing:
            raise IncidentError(
                f"Stored incident is missing {', '.join(missing)}"
            )

        # Fall back to the dataclass defaults for anything absent, so an
        # incident written by an older version still loads.
        defaults = {
            "subsystem": DEFAULT_SUBSYSTEM,
            "severity": DEFAULT_SEVERITY,
            "count": 1,
            "first_seen": "",
            "last_seen": "",
            "source_text": "",
            "source_offset": 0,
        }
        for name, value in defaults.items():
            if known.get(name) is None:
                known[name] = value

        return cls(**known)


@dataclass
class IncidentMatch:
    """An incident together with how similar it is to what was searched."""

    incident: FaultIncident
    similarity: float

    @property
    def message(self) -> str:
        return self.incident.message

    @property
    def subsystem(self) -> str:
        return self.incident.subsystem

    @property
    def severity(self) -> str:
        return self.incident.severity

    @property
    def error_code(self) -> str | None:
        return self.incident.error_code

    @property
    def count(self) -> int:
        return self.incident.count

    @property
    def first_seen(self) -> str:
        return self.incident.first_seen

    @property
    def last_seen(self) -> str:
        return self.incident.last_seen

    @property
    def previous_action(self) -> str | None:
        return self.incident.previous_action

    @property
    def outcome(self) -> str | None:
        return self.incident.outcome

    @property
    def source_text(self) -> str:
        return self.incident.source_text

    @property
    def offset(self) -> int:
        return self.incident.source_offset


@dataclass
class RecordFaultResult:
    """What a reported fault matched, and what became of it in storage.

    Callers get the storage outcome as data rather than having to read
    printed text or infer it from a length.
    """

    matches: list[IncidentMatch]
    stored: bool
    deduplicated: bool
    incident_id: int | None
    evicted_incident_id: int | None
    reason: str


def normalize_subsystem(subsystem: str | None) -> str:
    """Return a valid subsystem name, defaulting to UNKNOWN."""
    if subsystem is None or not str(subsystem).strip():
        return DEFAULT_SUBSYSTEM

    candidate = str(subsystem).strip().upper()

    if candidate not in SUBSYSTEMS:
        raise IncidentError(
            f"{subsystem!r} is not a known subsystem; expected one of "
            f"{', '.join(SUBSYSTEMS)}"
        )

    return candidate


def _codes_conflict(left: str | None, right: str | None) -> bool:
    """Return whether two error codes actively disagree.

    Two different codes are a conflict.  A code against no code is not:
    the older report simply did not carry one.
    """
    if not left or not right:
        return False

    return left != right


def find_duplicate(
    candidate: FaultIncident,
    incidents: list[FaultIncident],
    similarities: dict[int, float],
    dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> FaultIncident | None:
    """Return the incident ``candidate`` is a repeat of, if any.

    Two independent routes, both deliberately strict:

    Case A -- both carry the SAME non-empty error code and the same
    subsystem.  An error code is an explicit identity claim by whatever
    produced the fault, so it does not need a similarity check at all.

    Case B -- same subsystem, similarity at or above the dedup
    threshold, and no conflicting error code.

    Anything else is a new incident.  A wrong merge silently destroys
    the distinction between two faults forever, whereas a wrong split
    costs one embedding, so every uncertain case splits.
    """
    if candidate.error_code:
        for incident in incidents:
            if incident.error_code != candidate.error_code:
                continue

            # The same code from a different subsystem is exactly the
            # kind of coincidence that must NOT merge silently.
            if incident.subsystem == candidate.subsystem:
                return incident

    best: FaultIncident | None = None
    best_similarity = 0.0

    for incident in incidents:
        similarity = similarities.get(incident.incident_id)

        if similarity is None or similarity < dedup_threshold:
            continue

        if incident.subsystem != candidate.subsystem:
            continue

        if _codes_conflict(incident.error_code, candidate.error_code):
            continue

        if best is None or similarity > best_similarity:
            best = incident
            best_similarity = similarity

    return best
