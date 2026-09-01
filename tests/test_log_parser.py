"""Tests for reading the application log into structured records."""

from src.logsearch.log_parser import (
    FAULT_LEVELS,
    LogRecord,
    fault_records,
    parse_fault_records,
    parse_log_file,
    parse_log_lines,
    records_after,
)


SAMPLE_LOG = """2026-09-01 10:00:00 | INFO | Application started.
2026-09-01 10:00:05 | ERROR | Semantic search failed.
2026-09-01 10:00:09 | WARNING | Battery temperature is high.
2026-09-01 10:00:12 | INFO | Search completed successfully.
2026-09-01 10:00:20 | CRITICAL | Attitude control lost.
"""


def test_parses_timestamp_level_and_message(tmp_path):
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")

    assert records[0] == LogRecord(
        timestamp="2026-09-01 10:00:00",
        level="INFO",
        message="Application started.",
        source_text="autocomplete.log",
        offset=1,
    )


def test_offsets_are_one_based_line_numbers():
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")

    assert [record.offset for record in records] == [1, 2, 3, 4, 5]


def test_messages_containing_pipes_are_kept_whole():
    line = "2026-09-01 10:00:00 | ERROR | failed: a | b | c"

    records = parse_log_lines([line], "autocomplete.log")

    assert records[0].message == "failed: a | b | c"


def test_traceback_lines_belong_to_the_previous_record():
    lines = [
        "2026-09-01 10:00:00 | ERROR | Corpus preparation failed: boom",
        "Traceback (most recent call last):",
        '  File "main.py", line 43, in main',
        "    index = initialize_corpus(path)",
        "RuntimeError: boom",
        "2026-09-01 10:00:01 | INFO | Recovered.",
    ]

    records = parse_log_lines(lines, "autocomplete.log")

    assert len(records) == 2
    assert records[0].offset == 1
    assert records[0].message.startswith("Corpus preparation failed: boom")
    assert "RuntimeError: boom" in records[0].message
    assert records[0].message.count("\n") == 4
    assert records[1].offset == 6
    assert records[1].message == "Recovered."


def test_a_traceback_at_the_end_of_the_file_is_kept():
    lines = [
        "2026-09-01 10:00:00 | ERROR | It broke",
        "Traceback (most recent call last):",
        "ValueError: bad",
    ]

    records = parse_log_lines(lines, "autocomplete.log")

    assert len(records) == 1
    assert "ValueError: bad" in records[0].message


def test_text_before_the_first_entry_is_ignored():
    lines = [
        "leftover text with no header",
        "2026-09-01 10:00:00 | ERROR | It broke",
    ]

    records = parse_log_lines(lines, "autocomplete.log")

    assert len(records) == 1
    assert records[0].offset == 2
    assert records[0].message == "It broke"


def test_only_fault_levels_are_kept():
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")

    faults = fault_records(records)

    assert [record.level for record in faults] == [
        "ERROR",
        "WARNING",
        "CRITICAL",
    ]


def test_info_messages_are_never_treated_as_faults():
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")

    assert "INFO" not in {record.level for record in fault_records(records)}
    assert "INFO" not in FAULT_LEVELS


def test_fault_records_keep_their_original_offsets():
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")

    assert [record.offset for record in fault_records(records)] == [2, 3, 5]


def test_reads_a_real_file_and_defaults_the_source_name(tmp_path):
    log_path = tmp_path / "autocomplete.log"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")

    records = parse_fault_records(log_path)

    assert len(records) == 3
    assert {record.source_text for record in records} == {"autocomplete.log"}


def test_a_missing_file_yields_no_records(tmp_path):
    assert parse_log_file(tmp_path / "absent.log") == []


def test_an_empty_file_yields_no_records(tmp_path):
    log_path = tmp_path / "autocomplete.log"
    log_path.write_text("", encoding="utf-8")

    assert parse_log_file(log_path) == []


def test_records_after_selects_only_newer_lines():
    records = parse_log_lines(SAMPLE_LOG.splitlines(), "autocomplete.log")
    faults = fault_records(records)

    assert [record.offset for record in records_after(faults, 2)] == [3, 5]
    assert records_after(faults, 5) == []
    assert len(records_after(faults, 0)) == 3


def test_non_ascii_messages_survive(tmp_path):
    log_path = tmp_path / "autocomplete.log"
    log_path.write_text(
        "2026-09-01 10:00:00 | ERROR | Grüße — sensor offline\n",
        encoding="utf-8",
    )

    records = parse_fault_records(log_path)

    assert records[0].message == "Grüße — sensor offline"
