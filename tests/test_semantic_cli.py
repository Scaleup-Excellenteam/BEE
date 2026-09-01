"""Tests for the mode menu and the semantic log search CLI mode."""

import re
from dataclasses import dataclass
from unittest.mock import Mock, call

import pytest

import cli


@dataclass
class FakeSemanticResult:
    sentence: str
    source_text: str
    offset: int
    similarity: float


def fake_result(
    sentence="Semantic search failed.",
    offset=123,
    similarity=0.9124,
):
    """Return one result shaped like a SemanticResult."""
    return FakeSemanticResult(
        sentence=sentence,
        source_text="autocomplete.log",
        offset=offset,
        similarity=similarity,
    )


def test_menu_displays_all_modes_and_exit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", Mock(return_value="3"))

    cli.run_mode_menu()

    output = capsys.readouterr().out
    assert "SMART AUTOCOMPLETE" in output
    assert "1. Regular Autocomplete" in output
    assert "2. Semantic Log Search" in output
    assert "3. Exit" in output


def test_invalid_menu_option_allows_another_selection(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["invalid", "3"]),
    )

    cli.run_mode_menu()

    output = capsys.readouterr().out
    assert "Invalid option. Please choose 1, 2, or 3." in output
    assert output.count("SMART AUTOCOMPLETE") == 2


@pytest.mark.parametrize("termination", [EOFError(), KeyboardInterrupt()])
def test_menu_termination_is_clean(monkeypatch, capsys, termination):
    monkeypatch.setattr("builtins.input", Mock(side_effect=termination))

    cli.run_mode_menu()

    assert "Traceback" not in capsys.readouterr().out


# ----------------------------------------------------------------------
# Mode 1: Regular Autocomplete (Part A, unchanged)
# ----------------------------------------------------------------------


def test_mode_one_reuses_regular_autocomplete(monkeypatch):
    regular_autocomplete = Mock(return_value=True)
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["1", "3"]),
    )
    monkeypatch.setattr(cli, "main", regular_autocomplete)

    cli.run_mode_menu()

    regular_autocomplete.assert_called_once_with()


def test_mode_one_preserves_hash_reset_behavior(monkeypatch):
    read_prefilled_input = Mock(
        side_effect=[
            "previous sentence#",
            "new sentence",
            EOFError(),
        ]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr("builtins.input", Mock(return_value="1"))
    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.run_mode_menu()

    assert read_prefilled_input.call_args_list == [
        call(""),
        call(""),
        call("new sentence"),
    ]
    get_best_k_completions.assert_called_once_with("new sentence")


def test_mode_one_allows_several_searches_then_returns_to_the_menu(
    monkeypatch,
    capsys,
):
    read_prefilled_input = Mock(
        side_effect=["database", "python", "back"]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["1", "3"]),
    )
    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.run_mode_menu()

    assert get_best_k_completions.call_args_list == [
        call("database"),
        call("python"),
    ]

    output = capsys.readouterr().out
    assert "Regular Autocomplete" in output
    assert "Type 'back' to return to the main menu." in output
    # Menu shown once before the mode and once after "back".
    assert output.count("SMART AUTOCOMPLETE") == 2


def test_mode_one_never_sends_back_to_autocomplete(monkeypatch):
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["1", "3"]),
    )
    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        Mock(side_effect=["back"]),
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.run_mode_menu()

    get_best_k_completions.assert_not_called()


def test_mode_one_still_resets_on_hash_before_going_back(monkeypatch):
    read_prefilled_input = Mock(
        side_effect=["first sentence", "second#", "third", "back"]
    )
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["1", "3"]),
    )
    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        read_prefilled_input,
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.run_mode_menu()

    assert read_prefilled_input.call_args_list == [
        call(""),
        call("first sentence"),
        call(""),
        call("third"),
    ]
    assert get_best_k_completions.call_args_list == [
        call("first sentence"),
        call("third"),
    ]


def test_mode_one_can_be_re_entered_after_going_back(monkeypatch):
    get_best_k_completions = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["1", "1", "3"]),
    )
    monkeypatch.setattr(
        cli,
        "_read_prefilled_input",
        Mock(side_effect=["alpha", "back", "beta", "back"]),
    )
    monkeypatch.setattr(
        cli,
        "get_best_k_completions",
        get_best_k_completions,
    )

    cli.run_mode_menu()

    assert get_best_k_completions.call_args_list == [
        call("alpha"),
        call("beta"),
    ]


# ----------------------------------------------------------------------
# Mode 2: Semantic Log Search
# ----------------------------------------------------------------------


def test_mode_two_searches_the_log_and_displays_results(monkeypatch, capsys):
    log_search = Mock(return_value=[fake_result()])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "communication failure", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    log_search.assert_called_once_with("communication failure")

    output = capsys.readouterr().out
    assert "Semantic Log Search" in output
    assert "Type 'back' to return to the main menu." in output
    assert "Enter an error/message:" in output
    assert "Top 1 similar historical faults:" in output
    assert "1. Semantic search failed." in output
    assert "Source: autocomplete.log:123" in output
    assert "Similarity: 0.91" in output
    assert "score" not in output.lower()


def test_mode_two_displays_five_results_in_order(monkeypatch, capsys):
    results = [
        fake_result(
            f"Fault number {index}",
            offset=index,
            similarity=0.9 - index / 100,
        )
        for index in range(5)
    ]
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "failure", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=Mock(return_value=results))

    output = capsys.readouterr().out
    assert "Top 5 similar historical faults:" in output
    for index in range(5):
        assert f"{index + 1}. Fault number {index}" in output


def test_mode_two_allows_several_searches_then_returns_to_the_menu(
    monkeypatch,
    capsys,
):
    log_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "database", "authentication", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    assert log_search.call_args_list == [
        call("database"),
        call("authentication"),
    ]
    assert capsys.readouterr().out.count("SMART AUTOCOMPLETE") == 2


def test_mode_two_never_sends_back_to_the_log_search(monkeypatch):
    log_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    log_search.assert_not_called()


@pytest.mark.parametrize("typed", ["BACK", "  back  ", "Back"])
def test_back_is_recognized_regardless_of_case_and_spacing(monkeypatch, typed):
    log_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", typed, "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    log_search.assert_not_called()


def test_a_typed_query_is_only_a_search(monkeypatch):
    """A typed query must never be recorded as a real fault."""
    log_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "disk failure", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    # Exactly one read-only lookup, with nothing that could append.
    log_search.assert_called_once_with("disk failure")


def test_mode_two_reports_when_nothing_is_similar(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "unknown topic", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=Mock(return_value=[]))

    assert "No similar historical faults found." in capsys.readouterr().out


def test_mode_two_shortens_a_multi_line_entry(monkeypatch, capsys):
    traceback = (
        "Corpus preparation failed: boom\nTraceback:\nRuntimeError: boom"
    )
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "corpus", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[fake_result(traceback)])
    )

    output = capsys.readouterr().out
    assert "1. Corpus preparation failed: boom" in output
    assert "(2 more line(s) in this entry)" in output
    assert "RuntimeError: boom" not in output


def test_mode_two_survives_a_failed_query_and_keeps_asking(
    monkeypatch,
    capsys,
):
    log_search = Mock(side_effect=[RuntimeError("service failed"), []])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "broken", "recovered", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=log_search)

    assert log_search.call_count == 2
    output = capsys.readouterr().out
    assert "Semantic Log Search is temporarily unavailable." in output
    assert "Traceback" not in output


def test_unavailable_log_search_is_reported_and_menu_remains_usable(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "3"]),
    )

    cli.run_mode_menu()

    output = capsys.readouterr().out
    assert "Semantic Log Search is temporarily unavailable." in output
    assert output.count("SMART AUTOCOMPLETE") == 2


def test_mode_three_exits_without_invoking_either_search(monkeypatch):
    regular_autocomplete = Mock()
    log_search = Mock()
    monkeypatch.setattr("builtins.input", Mock(return_value="3"))
    monkeypatch.setattr(cli, "main", regular_autocomplete)

    cli.run_mode_menu(log_search_fn=log_search)

    regular_autocomplete.assert_not_called()
    log_search.assert_not_called()


@pytest.mark.parametrize("termination", [EOFError(), KeyboardInterrupt()])
def test_log_search_query_termination_is_clean(
    monkeypatch,
    capsys,
    termination,
):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", termination]),
    )

    cli.run_mode_menu(log_search_fn=Mock(return_value=[]))

    assert "Traceback" not in capsys.readouterr().out


def test_exiting_from_the_menu_never_enters_a_mode(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", Mock(side_effect=["3"]))

    cli.run_mode_menu(log_search_fn=Mock())

    output = capsys.readouterr().out
    assert "Regular Autocomplete\nType" not in output
    assert "Enter an error/message:" not in output


# ----------------------------------------------------------------------
# Query performance metrics
# ----------------------------------------------------------------------


def test_a_query_reports_its_cost(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "communication failure", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[fake_result(), fake_result()]),
        log_size_fn=Mock(return_value=1200),
    )

    output = capsys.readouterr().out
    assert "Query completed:" in output
    assert re.search(r"  Search time: \d+\.\d ms", output)
    assert "  Historical faults searched: 1200" in output
    assert "  Results returned: 2" in output


def test_the_reported_search_time_is_not_negative(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "failure", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[]),
        log_size_fn=Mock(return_value=10),
    )

    match = re.search(r"Search time: (\d+\.\d) ms", capsys.readouterr().out)
    assert match
    assert float(match.group(1)) >= 0.0


def test_metrics_are_reported_even_when_nothing_matches(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "unknown", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[]),
        log_size_fn=Mock(return_value=1200),
    )

    output = capsys.readouterr().out
    assert "No similar historical faults found." in output
    assert "  Results returned: 0" in output
    assert "  Historical faults searched: 1200" in output


def test_each_query_reports_its_own_metrics(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "first", "second", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[fake_result()]),
        log_size_fn=Mock(return_value=9),
    )

    assert capsys.readouterr().out.count("Query completed:") == 2


def test_the_corpus_size_is_omitted_when_it_is_unknown(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "failure", "back", "3"]),
    )

    cli.run_mode_menu(log_search_fn=Mock(return_value=[fake_result()]))

    output = capsys.readouterr().out
    assert "Query completed:" in output
    assert "Historical faults searched:" not in output
    assert "  Results returned: 1" in output


def test_a_failed_query_reports_no_metrics(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "broken", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(side_effect=RuntimeError("service failed")),
        log_size_fn=Mock(return_value=1200),
    )

    output = capsys.readouterr().out
    assert "Semantic Log Search is temporarily unavailable." in output
    assert "Query completed:" not in output


def test_the_size_callable_is_not_consulted_before_a_search(monkeypatch):
    log_size_fn = Mock(return_value=1200)
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[]),
        log_size_fn=log_size_fn,
    )

    log_size_fn.assert_not_called()


def test_query_metrics_are_logged_in_detail(monkeypatch, tmp_path):
    from src.logging_config import configure_logging, shutdown_logging

    log_path = tmp_path / "autocomplete.log"
    configure_logging(log_path)

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "disk failure", "back", "3"]),
    )

    cli.run_mode_menu(
        log_search_fn=Mock(return_value=[fake_result()]),
        log_size_fn=Mock(return_value=1200),
    )
    shutdown_logging()

    log_text = log_path.read_text(encoding="utf-8")
    assert re.search(
        r'Semantic log query "disk failure" completed in \d+\.\d ms over '
        r"1200 historical fault records, returning 1 results\.",
        log_text,
    )
