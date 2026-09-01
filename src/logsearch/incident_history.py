"""Canonical, persistent store of satellite fault incidents.

    logs/satellite_incidents.jsonl     one JSON object per incident

This file is the SOURCE OF TRUTH.  The vector cache beside it is an
optimization: if the cache is deleted, every incident can be re-embedded
from here and nothing is lost.  If this file is deleted, the history is
gone, cache or no cache.

Why JSONL and not the append-only log used before
-------------------------------------------------

An incident is mutable: each recurrence bumps ``count`` and
``last_seen``.  An append-only log cannot express that without either
rewriting history or storing one line per occurrence, and one line per
occurrence is exactly what incidents exist to avoid.

Writes rewrite the whole file through a temporary file and one atomic
replace, so a crash mid-write leaves the previous version intact rather
than a half-written one.  At incident scale -- thousands, not millions,
because recurrences collapse -- rewriting is a few hundred kilobytes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.logsearch.incidents import FaultIncident, IncidentError


DEFAULT_INCIDENTS_PATH = "logs/satellite_incidents.jsonl"


class IncidentHistoryError(RuntimeError):
    """Raised when the incident history cannot be read or written."""


class IncidentHistory:
    """Read and write the incident history file."""

    def __init__(self, path: str = DEFAULT_INCIDENTS_PATH):
        self.path = Path(path)

    def exists(self) -> bool:
        """Return whether a history file has been written yet."""
        return self.path.is_file()

    def load(self) -> list[FaultIncident]:
        """Return every stored incident, in id order."""
        if not self.exists():
            return []

        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise IncidentHistoryError(
                f"Unreadable incident history: {error}"
            ) from error

        incidents = []

        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise IncidentHistoryError(
                    f"Corrupted incident history on line {number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise IncidentHistoryError(
                    f"Incident history line {number} is not a JSON object"
                )

            try:
                incidents.append(FaultIncident.from_dict(record))
            except IncidentError as error:
                raise IncidentHistoryError(
                    f"Invalid incident on history line {number}: {error}"
                ) from error

        # Ids must ascend and never repeat, but eviction leaves gaps, so
        # a gapless 1..N sequence is NOT required.
        ids = [incident.incident_id for incident in incidents]

        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise IncidentHistoryError(
                "Incident history ids are not strictly increasing; the file "
                "has been reordered or edited"
            )

        return incidents

    @property
    def meta_path(self) -> Path:
        """Return the small side file holding the next incident id."""
        return self.path.with_suffix(".meta.json")

    def next_incident_id(self, incidents: list[FaultIncident]) -> int:
        """Return the id the next new incident must take.

        Ids are monotonic and never reused: after incident 2 is evicted
        from 1, 2, 3 the next incident is 4, not 2.  The counter lives
        beside the history because the highest surviving id is not enough
        on its own -- evicting the newest incident would otherwise hand
        its id straight back out.
        """
        highest_in_use = max(
            (one.incident_id for one in incidents),
            default=0,
        )

        try:
            stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
            recorded = stored.get("next_incident_id")
        except (OSError, json.JSONDecodeError, AttributeError):
            recorded = None

        if not isinstance(recorded, int):
            recorded = 0

        # Whichever is larger wins, so a lost or stale side file can only
        # ever waste a few ids, never cause one to be reused.
        return max(recorded, highest_in_use + 1)

    def save(
        self,
        incidents: list[FaultIncident],
        next_incident_id: int | None = None,
    ) -> None:
        """Write every incident, replacing the file atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # The id counter is written FIRST.  A crash between the two
        # writes then leaves the counter ahead of the history, which
        # wastes an id; the other order would re-issue one.
        if next_incident_id is not None:
            self._write_meta(next_incident_id)

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")

        with temporary.open("w", encoding="utf-8") as history_file:
            for incident in incidents:
                history_file.write(
                    json.dumps(incident.to_dict(), ensure_ascii=False) + "\n"
                )

        os.replace(temporary, self.path)

    def _write_meta(self, next_incident_id: int) -> None:
        """Record the next id to hand out, replacing the file atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.meta_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"next_incident_id": next_incident_id}, indent=2)
            + "\n",
            encoding="utf-8",
        )

        os.replace(temporary, self.meta_path)
