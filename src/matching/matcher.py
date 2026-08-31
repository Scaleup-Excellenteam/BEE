"""Internal matching helpers for normalized query and target strings."""

from src.matching.scoring import exact_match_score, substitution_score


def _score_equal_length_match(query: str, target: str) -> int | None:
    """Score an equal-length exact or one-substitution match."""
    mismatch_position = None

    for position, (query_character, target_character) in enumerate(
        zip(query, target),
        start=1,
    ):
        if query_character == target_character:
            continue

        if mismatch_position is not None:
            return None

        mismatch_position = position

    if mismatch_position is None:
        return exact_match_score(len(query))

    return substitution_score(len(query), mismatch_position)
