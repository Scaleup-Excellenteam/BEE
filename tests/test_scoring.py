import pytest

from src.matching.penalties import (
    insertion_deletion_penalty,
    substitution_penalty,
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
