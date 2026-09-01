"""Pure vector-similarity calculations for semantic search."""

import math


def _validate_vector(vector: list[float], name: str) -> tuple[list[float], float]:
    """Return numeric values and magnitude for a valid embedding vector."""
    if not vector:
        raise ValueError(f"{name} must not be empty")

    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values") from error

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")

    magnitude_squared = math.fsum(value * value for value in values)
    if magnitude_squared == 0.0:
        raise ValueError(f"{name} must have non-zero magnitude")
    if not math.isfinite(magnitude_squared):
        raise ValueError(f"{name} magnitude is not finite")

    return values, math.sqrt(magnitude_squared)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Return the cosine similarity between two valid embedding vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same dimensions")

    values_a, magnitude_a = _validate_vector(vector_a, "vector_a")
    values_b, magnitude_b = _validate_vector(vector_b, "vector_b")

    dot_product = math.fsum(
        value_a * value_b
        for value_a, value_b in zip(values_a, values_b)
    )
    if not math.isfinite(dot_product):
        raise ValueError("vector dot product is not finite")

    return dot_product / (magnitude_a * magnitude_b)
