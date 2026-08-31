import pytest

from src.matching.penalties import (
    insertion_deletion_penalty,
    substitution_penalty,
)
from src.matching.scoring import (
    exact_match_score,
    extra_character_score,
    missing_character_score,
    substitution_score,
)


@pytest.mark.parametrize(
    ("position", "expected_penalty"),
    [
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 2),
        (5, 1),
        (6, 1),
        (10, 1),
    ],
)
def test_substitution_penalty(position: int, expected_penalty: int) -> None:
    assert substitution_penalty(position) == expected_penalty


@pytest.mark.parametrize(
    ("position", "expected_penalty"),
    [
        (1, 10),
        (2, 8),
        (3, 6),
        (4, 4),
        (5, 2),
        (6, 2),
        (10, 2),
    ],
)
def test_insertion_deletion_penalty(
    position: int,
    expected_penalty: int,
) -> None:
    assert insertion_deletion_penalty(position) == expected_penalty


def test_exact_match_score() -> None:
    assert exact_match_score(query_length=5) == 10


@pytest.mark.parametrize(
    ("position", "expected_score"),
    [
        (1, 3),
        (2, 4),
        (4, 6),
        (5, 7),
        (8, 7),
    ],
)
def test_substitution_score(position: int, expected_score: int) -> None:
    assert substitution_score(query_length=5, position=position) == expected_score


@pytest.mark.parametrize(
    ("position", "expected_score"),
    [
        (1, -2),
        (2, 0),
        (4, 4),
        (5, 6),
        (8, 6),
    ],
)
def test_extra_character_score(position: int, expected_score: int) -> None:
    assert extra_character_score(query_length=5, position=position) == expected_score


@pytest.mark.parametrize(
    ("position", "expected_score"),
    [
        (1, 0),
        (2, 2),
        (4, 6),
        (5, 8),
        (8, 8),
    ],
)
def test_missing_character_score(position: int, expected_score: int) -> None:
    assert missing_character_score(query_length=5, position=position) == expected_score


def test_zero_score_remains_valid() -> None:
    score = extra_character_score(query_length=5, position=2)

    assert score == 0
    assert score is not None


def test_negative_score_remains_valid() -> None:
    score = extra_character_score(query_length=2, position=1)

    assert score == -8
    assert score is not None
