"""Pure score calculations for legal exact and one-edit matches."""

from src.matching.penalties import (
    insertion_deletion_penalty,
    substitution_penalty,
)


def exact_match_score(query_length: int) -> int:
    """Return the score for an exact match."""
    return 2 * query_length


def substitution_score(query_length: int, position: int) -> int:
    """Return the score for one substitution at a valid position."""
    matching_characters = query_length - 1
    return 2 * matching_characters - substitution_penalty(position)


def extra_character_score(query_length: int, position: int) -> int:
    """Return the score when the query contains one extra character."""
    matching_characters = query_length - 1
    return 2 * matching_characters - insertion_deletion_penalty(position)


def missing_character_score(query_length: int, position: int) -> int:
    """Return the score when the query is missing one character."""
    matching_characters = query_length
    return 2 * matching_characters - insertion_deletion_penalty(position)
