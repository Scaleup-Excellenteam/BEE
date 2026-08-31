import pytest

import src.matching.matcher as matcher_module
from src.matching.matcher import (
    _score_equal_length_match,
    _score_extra_character_match,
    _score_missing_character_match,
    calculate_best_match,
)


def test_exact_equal_length_match() -> None:
    assert _score_equal_length_match("to be", "to be") == 10


@pytest.mark.parametrize(
    ("query", "target", "expected_score"),
    [
        ("abcde", "xbcde", 3),
        ("abcde", "axcde", 4),
        ("abcde", "abcxe", 6),
        ("abcde", "abcdx", 7),
        ("abcdef", "abcdex", 9),
    ],
)
def test_one_substitution(
    query: str,
    target: str,
    expected_score: int,
) -> None:
    assert _score_equal_length_match(query, target) == expected_score


def test_space_is_counted_in_substitution_position() -> None:
    assert _score_equal_length_match("to be", "to bx") == 7


def test_space_can_be_the_substituted_character() -> None:
    assert _score_equal_length_match("to be", "to-be") == 5


def test_two_mismatches_returns_none() -> None:
    assert _score_equal_length_match("abcde", "abxye") is None


def test_more_than_two_mismatches_returns_none() -> None:
    assert _score_equal_length_match("abcde", "axxyz") is None


@pytest.mark.parametrize(
    ("query", "target", "expected_score"),
    [
        ("xabc", "abc", -4),
        ("abxc", "abc", 0),
        ("abcx", "abc", 2),
    ],
)
def test_extra_character_match(
    query: str,
    target: str,
    expected_score: int,
) -> None:
    assert _score_extra_character_match(query, target) == expected_score


def test_extra_character_position_counts_spaces() -> None:
    assert _score_extra_character_match("to xbe", "to be") == 6


def test_extra_character_match_rejects_second_mismatch() -> None:
    assert _score_extra_character_match("abxd", "acd") is None


def test_extra_repeated_character_uses_highest_scoring_alignment() -> None:
    assert _score_extra_character_match("aaab", "aab") == 0


@pytest.mark.parametrize(
    ("query", "target", "expected_score"),
    [
        ("abc", "xabc", -4),
        ("abc", "abxc", 0),
        ("abc", "abcx", 2),
    ],
)
def test_missing_character_match(
    query: str,
    target: str,
    expected_score: int,
) -> None:
    assert _score_missing_character_match(query, target) == expected_score


def test_missing_character_position_counts_spaces() -> None:
    assert _score_missing_character_match("tobe", "to be") == 2


def test_missing_character_match_rejects_second_mismatch() -> None:
    assert _score_missing_character_match("acd", "abxd") is None


def test_missing_repeated_character_uses_highest_scoring_alignment() -> None:
    assert _score_missing_character_match("aab", "aaab") == 0


@pytest.mark.parametrize(
    ("query", "sentence"),
    [
        ("abc", "abcdef"),
        ("abc", "xxabcxx"),
        ("abc", "xxabc"),
    ],
)
def test_calculate_best_match_finds_exact_substring(
    query: str,
    sentence: str,
) -> None:
    assert calculate_best_match(query, sentence) == 6


def test_calculate_best_match_finds_substitution_inside_sentence() -> None:
    assert calculate_best_match("abcde", "zzabxdezz") == 5


def test_calculate_best_match_finds_extra_query_character() -> None:
    assert calculate_best_match("abxcd", "zzabcdzz") == 2


def test_calculate_best_match_finds_missing_query_character() -> None:
    assert calculate_best_match("abcd", "zzabxcdzz") == 2


def test_calculate_best_match_rejects_more_than_one_edit() -> None:
    assert calculate_best_match("hxlpo", "hello") is None


def test_calculate_best_match_skips_windows_longer_than_sentence() -> None:
    assert calculate_best_match("abcdef", "abc") is None


def test_calculate_best_match_returns_highest_approximate_score() -> None:
    assert calculate_best_match("abc", "xbc ayc") == 0


def test_calculate_best_match_preserves_zero_score() -> None:
    score = calculate_best_match("abxc", "abc")

    assert score == 0
    assert score is not None


def test_calculate_best_match_preserves_negative_score() -> None:
    score = calculate_best_match("xabc", "abc")

    assert score == -4
    assert score is not None


def test_calculate_best_match_rejects_empty_query() -> None:
    assert calculate_best_match("", "abc") is None


def test_calculate_best_match_rejects_empty_sentence() -> None:
    assert calculate_best_match("abc", "") is None


def test_calculate_best_match_preserves_repeated_character_alignment() -> None:
    assert calculate_best_match("aaab", "aab") == 0


def test_one_character_query_does_not_scan_zero_length_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(query: str, target: str) -> int | None:
        raise AssertionError("zero-length target window was scanned")

    monkeypatch.setattr(
        matcher_module,
        "_score_extra_character_match",
        fail_if_called,
    )

    assert calculate_best_match("a", "x") == -5


@pytest.mark.parametrize(
    ("query", "expected_score"),
    [
        ("to be", 10),
        ("or not", 12),
        ("be that", 14),
        ("2o be", 3),
        ("to pe", 6),
        ("or knot", 8),
        ("or nt", 8),
        ("not be", None),
    ],
)
def test_official_golden_matches(
    query: str,
    expected_score: int | None,
) -> None:
    sentence = "to be or not to be that is the question"

    assert calculate_best_match(query, sentence) == expected_score
