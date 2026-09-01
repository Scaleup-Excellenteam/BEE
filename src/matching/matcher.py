"""Match normalized queries against normalized candidate sentences."""

from src.matching.scoring import (
    exact_match_score,
    extra_character_score,
    missing_character_score,
    substitution_score,
)


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


def _score_extra_character_match(query: str, target: str) -> int | None:
    """Score a match where the query contains exactly one extra character."""
    query_index = 0
    target_index = 0
    extra_position = None

    while query_index < len(query) and target_index < len(target):
        if query[query_index] == target[target_index]:
            query_index += 1
            target_index += 1
            continue

        if extra_position is not None:
            return None

        extra_position = query_index + 1
        query_index += 1

    if extra_position is None:
        extra_position = query_index + 1

    return extra_character_score(len(query), extra_position)


def _score_missing_character_match(query: str, target: str) -> int | None:
    """Score a match where the query is missing exactly one character."""
    query_index = 0
    target_index = 0
    missing_position = None

    while query_index < len(query) and target_index < len(target):
        if query[query_index] == target[target_index]:
            query_index += 1
            target_index += 1
            continue

        if missing_position is not None:
            return None

        missing_position = query_index + 1
        target_index += 1

    if missing_position is None:
        missing_position = query_index + 1

    return missing_character_score(len(query), missing_position)


def calculate_best_match(query: str, sentence: str) -> int | None:
    """Return the highest score for a legal substring match, or ``None``."""
    if query == "" or sentence == "":
        return None

    query_length = len(query)
    exact_score = exact_match_score(query_length)
    window_scorers = [
        (query_length, _score_equal_length_match),
    ]

    # Temporary team decision: a one-character query does not use empty windows.
    if query_length - 1 > 0:
        window_scorers.append(
            (query_length - 1, _score_extra_character_match),
        )

    window_scorers.append(
        (query_length + 1, _score_missing_character_match),
    )

    best_score = None

    for window_length, score_window in window_scorers:
        if window_length > len(sentence):
            continue

        for start in range(len(sentence) - window_length + 1):
            target = sentence[start : start + window_length]
            score = score_window(query, target)

            if score is None:
                continue

            if window_length == query_length and score == exact_score:
                return score

            if best_score is None or score > best_score:
                best_score = score

    return best_score
