import pytest

from src.matching.matcher import _score_equal_length_match


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
