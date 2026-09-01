"""Position-based penalties for one-edit matches."""


def substitution_penalty(position: int) -> int:
    """Return the substitution penalty for a valid 1-based position."""
    return max(6 - position, 1)


def insertion_deletion_penalty(position: int) -> int:
    """Return the insertion/deletion penalty for a valid 1-based position."""
    return max(12 - 2 * position, 2)
