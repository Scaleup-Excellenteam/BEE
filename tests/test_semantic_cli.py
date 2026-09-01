"""Tests for the dependency-injected semantic CLI mode."""

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


def test_menu_displays_all_modes_and_exit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", Mock(return_value="3"))

    cli.run_mode_menu()

    output = capsys.readouterr().out
    assert "SMART AUTOCOMPLETE" in output
    assert "1. Regular Autocomplete" in output
    assert "2. Semantic Search" in output
    assert "3. Exit" in output


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


def test_mode_two_invokes_search_and_displays_result(monkeypatch, capsys):
    inputs = Mock(side_effect=["2", "how do I login", "back", "3"])
    embedded_sentences = [object()]
    embedder = Mock()
    semantic_search = Mock(
        return_value=[
            FakeSemanticResult(
                sentence=(
                    "Authentication is required before accessing the system."
                ),
                source_text="sg244986.txt",
                offset=17425,
                similarity=0.9124,
            )
        ]
    )

    monkeypatch.setattr("builtins.input", inputs)

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=embedded_sentences,
        embedder=embedder,
    )

    semantic_search.assert_called_once_with(
        "how do I login",
        embedded_sentences,
        embedder,
    )
    output = capsys.readouterr().out
    assert "Here are 1 semantic results:" in output
    assert (
        "1. Authentication is required before accessing the system."
        in output
    )
    assert "Source: sg244986.txt:17425" in output
    assert "Similarity: 0.912" in output
    assert "score" not in output.lower()


def test_empty_semantic_query_is_passed_to_search(monkeypatch, capsys):
    semantic_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "   ", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[],
        embedder=Mock(),
    )

    assert semantic_search.call_args.args[0] == "   "
    assert "No semantic results found." in capsys.readouterr().out


def test_semantic_no_results_message(monkeypatch, capsys):
    semantic_search = Mock(return_value=[])
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "unknown topic", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    assert "No semantic results found." in capsys.readouterr().out


def test_semantic_failure_is_handled_and_menu_remains_usable(
    monkeypatch,
    capsys,
):
    semantic_search = Mock(side_effect=RuntimeError("service failed"))
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "query", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    output = capsys.readouterr().out
    assert "Semantic Search is temporarily unavailable." in output
    assert "Traceback" not in output
    assert output.count("SMART AUTOCOMPLETE") == 2


def test_missing_semantic_dependencies_are_handled(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "3"]),
    )

    cli.run_mode_menu()

    assert (
        "Semantic Search is temporarily unavailable."
        in capsys.readouterr().out
    )


def test_invalid_menu_option_allows_another_selection(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["invalid", "3"]),
    )

    cli.run_mode_menu()

    output = capsys.readouterr().out
    assert "Invalid option. Please choose 1, 2, or 3." in output
    assert output.count("SMART AUTOCOMPLETE") == 2


def test_mode_three_exits_without_invoking_either_search(monkeypatch):
    regular_autocomplete = Mock()
    semantic_search = Mock()
    monkeypatch.setattr("builtins.input", Mock(return_value="3"))
    monkeypatch.setattr(cli, "main", regular_autocomplete)

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    regular_autocomplete.assert_not_called()
    semantic_search.assert_not_called()


@pytest.mark.parametrize("termination", [EOFError(), KeyboardInterrupt()])
def test_menu_termination_is_clean(monkeypatch, capsys, termination):
    monkeypatch.setattr("builtins.input", Mock(side_effect=termination))

    cli.run_mode_menu()

    assert "Traceback" not in capsys.readouterr().out


@pytest.mark.parametrize("termination", [EOFError(), KeyboardInterrupt()])
def test_semantic_query_termination_is_clean(
    monkeypatch,
    capsys,
    termination,
):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", termination]),
    )

    cli.run_mode_menu(
        semantic_search_fn=Mock(),
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    assert "Traceback" not in capsys.readouterr().out


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


def test_mode_two_allows_several_searches_then_returns_to_the_menu(
    monkeypatch,
    capsys,
):
    semantic_search = Mock(return_value=[])
    embedded_sentences = [object()]
    embedder = Mock()

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "database", "authentication", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=embedded_sentences,
        embedder=embedder,
    )

    assert semantic_search.call_args_list == [
        call("database", embedded_sentences, embedder),
        call("authentication", embedded_sentences, embedder),
    ]

    output = capsys.readouterr().out
    assert "Type 'back' to return to the main menu." in output
    assert output.count("SMART AUTOCOMPLETE") == 2


def test_mode_two_never_sends_back_to_semantic_search(monkeypatch):
    semantic_search = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    semantic_search.assert_not_called()


@pytest.mark.parametrize("typed", ["BACK", "  back  ", "Back"])
def test_back_is_recognized_regardless_of_case_and_spacing(
    monkeypatch,
    typed,
):
    semantic_search = Mock(return_value=[])

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", typed, "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    semantic_search.assert_not_called()


def test_mode_two_survives_a_failed_query_and_keeps_asking(
    monkeypatch,
    capsys,
):
    semantic_search = Mock(
        side_effect=[RuntimeError("service failed"), []]
    )

    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=["2", "broken", "recovered", "back", "3"]),
    )

    cli.run_mode_menu(
        semantic_search_fn=semantic_search,
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    assert semantic_search.call_count == 2
    output = capsys.readouterr().out
    assert "Semantic Search is temporarily unavailable." in output
    assert "No semantic results found." in output


def test_exiting_from_the_menu_never_enters_a_mode(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", Mock(side_effect=["3"]))

    cli.run_mode_menu(
        semantic_search_fn=Mock(),
        embedded_sentences=[object()],
        embedder=Mock(),
    )

    output = capsys.readouterr().out
    assert "Regular Autocomplete\nType" not in output
    assert "Enter your query:" not in output
