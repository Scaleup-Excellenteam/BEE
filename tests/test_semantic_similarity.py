"""Tests for pure semantic vector similarity."""

import math

import pytest

from src.semantic.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_known_non_trivial_similarity():
    expected = 11.0 / math.sqrt(125.0)

    assert cosine_similarity([1.0, 2.0], [3.0, 4.0]) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("vector_a", "vector_b"),
    [
        ([], []),
        ([], [1.0]),
        ([1.0], []),
    ],
)
def test_empty_vectors_raise_value_error(vector_a, vector_b):
    with pytest.raises(ValueError):
        cosine_similarity(vector_a, vector_b)


def test_different_dimensions_raise_value_error():
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity([1.0, 0.0], [1.0])


@pytest.mark.parametrize(
    ("vector_a", "vector_b"),
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
    ],
)
def test_zero_vector_raises_value_error(vector_a, vector_b):
    with pytest.raises(ValueError, match="non-zero magnitude"):
        cosine_similarity(vector_a, vector_b)


@pytest.mark.parametrize("invalid_value", [math.inf, -math.inf, math.nan])
def test_non_finite_vector_raises_value_error(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([invalid_value], [1.0])
