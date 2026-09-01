"""Read the application log file into structured records.

The log is written by ``src/logging_config.py`` with the format

    %(asctime)s | %(levelname)s | %(message)s

so a normal entry is one line of three pipe separated fields:

    2026-09-01 11:05:42 | ERROR | Semantic search failed.

``LOGGER.exception`` also writes a traceback underneath its own header
line.  Those continuation lines are NOT records of their own: they belong
to the entry above them, and they are folded back into that entry's
message so the traceback text is searchable together with the message
that introduced it.

The log file is opened in append mode and is never rewritten, so a line
number is a permanent identifier for a record.  That is what makes
incremental indexing possible: anything past the last indexed line is
new.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Timestamp | LEVEL | message.  Anything else is a continuation line.
LOG_ENTRY_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r" \| (?P<level>[A-Z]+)"
    r" \| (?P<message>.*)$"
)

# Only faults are worth searching.  Ordinary INFO chatter ("Application
# started", "Search completed successfully") would swamp the results.
FAULT_LEVELS = frozenset({"ERROR", "WARNING", "CRITICAL"})


@dataclass(frozen=True)
class LogRecord:
    """One log entry, including any traceback lines beneath it."""

    timestamp: str
    level: str
    message: str
    source_text: str
    offset: int


def parse_log_lines(
    lines: Iterable[str],
    source_text: str,
) -> list[LogRecord]:
    """Return every log entry found in ``lines``.

    ``offset`` is the 1 based number of the line the entry starts on.
    """
    records: list[LogRecord] = []
    pending_match = None
    pending_offset = 0
    continuation: list[str] = []

    def flush() -> None:
        """Turn the entry being accumulated into a record."""
        if pending_match is None:
            return

        message = pending_match.group("message")

        if continuation:
            message = "\n".join([message, *continuation])

        records.append(
            LogRecord(
                timestamp=pending_match.group("timestamp"),
                level=pending_match.group("level"),
                message=message,
                source_text=source_text,
                offset=pending_offset,
            )
        )

    for number, line in enumerate(lines, start=1):
        line = line.rstrip("\n").rstrip("\r")
        match = LOG_ENTRY_PATTERN.match(line)

        if match is None:
            # A traceback line, or anything else that is not a header.
            # It belongs to the entry above it; text before the first
            # entry has no owner and is dropped.
            if pending_match is not None:
                continuation.append(line)

            continue

        flush()
        pending_match = match
        pending_offset = number
        continuation = []

    flush()

    return records


def parse_log_file(path: str | Path, source_text: str = "") -> list[LogRecord]:
    """Return every log entry in the file at ``path``.

    ``source_text`` defaults to the file's own name, which is what ends
    up on each search result.
    """
    log_path = Path(path)

    if not log_path.is_file():
        return []

    text = log_path.read_text(encoding="utf-8", errors="replace")

    return parse_log_lines(
        text.splitlines(),
        source_text or log_path.name,
    )


def fault_records(records: Iterable[LogRecord]) -> list[LogRecord]:
    """Return only the records that report a fault."""
    return [record for record in records if record.level in FAULT_LEVELS]


def parse_fault_records(
    path: str | Path,
    source_text: str = "",
) -> list[LogRecord]:
    """Return the fault entries of the log file at ``path``."""
    return fault_records(parse_log_file(path, source_text))


def records_after(
    records: Iterable[LogRecord],
    offset: int,
) -> list[LogRecord]:
    """Return the records that start after line ``offset``."""
    return [record for record in records if record.offset > offset]
