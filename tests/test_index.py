"""Tests for the trigram candidate index.

The property under test is one-sided: ``get_candidates`` is allowed to return
extra sentences, but it must never lose one that could legally match.
"""

import random

from src.corpus.index import CorpusIndex
from src.models import SentenceRecord


def make_records(*sentences: str) -> list[SentenceRecord]:
    """Build a tiny synthetic corpus of already-normalized sentences."""
    return [
        SentenceRecord(
            original_sentence=sentence,
            normalized_sentence=sentence,
            source_text="synthetic.txt",
            offset=offset,
        )
        for offset, sentence in enumerate(sentences, start=1)
    ]


def normalized_results(records: list[SentenceRecord]) -> set[str]:
    return {record.normalized_sentence for record in records}


# ---------------------------------------------------------------------------
# TEST-ONLY reference matcher.
#
# This is a deliberately slow, obviously-correct brute force used only to
# prove that the index loses nothing.  It is NOT the production matcher and
# must never be moved into src/ - final matching and scoring belong to the
# matching stage, not to the index.
# ---------------------------------------------------------------------------

def _within_one_edit(left: str, right: str) -> bool:
    """True when ``left`` and ``right`` differ by at most one edit."""
    if left == right:
        return True

    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        differences = sum(1 for a, b in zip(left, right) if a != b)
        return differences == 1

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)

    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1

    return shorter[index:] == longer[index + 1:]


def _reference_could_match(query: str, sentence: str) -> bool:
    """True when some substring of ``sentence`` is within one edit of ``query``.

    This is exactly the set of sentences the index is forbidden to drop.
    """
    if not query:
        return False

    for length in (len(query) - 1, len(query), len(query) + 1):
        if length < 1:
            continue
        for start in range(len(sentence) - length + 1):
            if _within_one_edit(query, sentence[start:start + length]):
                return True

    return False


# ---------------------------------------------------------------------------
# Required behaviours
# ---------------------------------------------------------------------------

def test_exact_substring_is_returned():
    index = CorpusIndex(make_records("hello", "world"))

    assert "hello" in normalized_results(index.get_candidates("llo"))


def test_one_edit_query_is_not_filtered_out():
    # "helo" is "hello" with one character missing from the query.
    index = CorpusIndex(make_records("hello", "world"))

    assert "hello" in normalized_results(index.get_candidates("helo"))


def test_query_starting_mid_word_is_returned():
    index = CorpusIndex(make_records("computer science", "biology"))

    assert "computer science" in normalized_results(
        index.get_candidates("puter sci")
    )


def test_trigrams_cross_word_boundaries():
    index = CorpusIndex(make_records("computer science", "biology"))

    # "er sci" only exists as a trigram sequence if spaces are real
    # characters and words are not split.
    assert "computer science" in normalized_results(
        index.get_candidates("uter science")
    )


def test_short_query_falls_back_to_every_record():
    records = make_records("hello", "world", "completely unrelated")
    index = CorpusIndex(records)

    # "abcd" has 2 distinct trigrams, which is <= 3, so nothing may be cut.
    assert index.get_candidates("abcd") == records
    assert index.get_candidates("ab") == records


def test_fallback_returns_a_copy_not_the_internal_list():
    records = make_records("hello", "world")
    index = CorpusIndex(records)

    returned = index.get_candidates("ab")
    returned.clear()

    assert len(index.records) == 2


def test_empty_query_returns_empty_list():
    index = CorpusIndex(make_records("hello", "world"))

    assert index.get_candidates("") == []


def test_repeated_trigram_does_not_duplicate_a_sentence():
    sentence = "the cat and the dog and the bird sat"
    index = CorpusIndex(make_records(sentence, "unrelated text here"))

    candidates = index.get_candidates("and the dog and the bird")

    matching = [
        record for record in candidates
        if record.normalized_sentence == sentence
    ]
    assert len(matching) == 1

    # The posting list itself must hold the sentence id only once, even
    # though "the" occurs three times in that sentence.
    assert list(index._postings["the"]).count(0) == 1


def test_results_are_sentence_record_objects():
    index = CorpusIndex(make_records("computer science", "hello world"))

    candidates = index.get_candidates("computer sci")

    assert candidates
    assert all(isinstance(record, SentenceRecord) for record in candidates)


def test_no_duplicate_records_are_returned():
    index = CorpusIndex(
        make_records("computer science", "computer graphics", "hello world")
    )

    candidates = index.get_candidates("computer sci")

    assert len(candidates) == len({id(record) for record in candidates})


def test_candidates_point_into_the_original_records():
    records = make_records("computer science", "hello world")
    index = CorpusIndex(records)

    candidates = index.get_candidates("computer sci")

    # Identity, not equality: no record was copied into the index.
    assert all(any(record is original for original in records)
               for record in candidates)


def test_index_actually_filters_something():
    # Guards against a trivially-correct implementation that returns
    # everything for every query.
    records = make_records(
        "computer science department",
        "the quick brown fox jumps",
    )
    index = CorpusIndex(records)

    candidates = index.get_candidates("computer science")

    assert normalized_results(candidates) == {"computer science department"}


def test_stats_report_the_built_index():
    records = make_records("hello world", "computer science")
    index = CorpusIndex(records)

    stats = index.stats()

    assert stats["record_count"] == 2
    assert stats["distinct_trigrams"] > 0
    assert stats["total_postings"] > 0
    assert stats["build_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# No-false-negative property, checked against the reference matcher
# ---------------------------------------------------------------------------

SMALL_CORPUS = [
    "hello",
    "hello world",
    "computer science",
    "computer graphics department",
    "the quick brown fox jumps over the lazy dog",
    "the cat and the dog and the bird sat",
    "python programming language",
    "index the corpus offline",
    "abcdefghij",
    "aaaaaaaaaaaa",
    "banana bandana",
    "",
]


def _assert_no_false_negatives(index, queries):
    for query in queries:
        candidates = normalized_results(index.get_candidates(query))
        for sentence in SMALL_CORPUS:
            if _reference_could_match(query, sentence):
                assert sentence in candidates, (
                    f"query {query!r} lost sentence {sentence!r}"
                )


def test_no_false_negatives_for_every_substring_query():
    index = CorpusIndex(make_records(*SMALL_CORPUS))

    queries = set()
    for sentence in SMALL_CORPUS:
        for start in range(len(sentence)):
            for length in range(1, 13):
                if start + length <= len(sentence):
                    queries.add(sentence[start:start + length])

    _assert_no_false_negatives(index, queries)


def test_no_false_negatives_for_mutated_queries():
    index = CorpusIndex(make_records(*SMALL_CORPUS))
    generator = random.Random(20240831)
    alphabet = "abcdefghijklmnopqrstuvwxyz "

    queries = set()
    for _ in range(400):
        sentence = generator.choice([s for s in SMALL_CORPUS if s])
        length = generator.randint(1, min(14, len(sentence)))
        start = generator.randint(0, len(sentence) - length)
        query = sentence[start:start + length]

        operation = generator.choice(["keep", "substitute", "delete", "insert"])
        position = generator.randint(0, max(0, len(query) - 1))
        replacement = generator.choice(alphabet)

        if operation == "substitute" and query:
            query = query[:position] + replacement + query[position + 1:]
        elif operation == "delete" and query:
            query = query[:position] + query[position + 1:]
        elif operation == "insert":
            query = query[:position] + replacement + query[position:]

        queries.add(query)

    _assert_no_false_negatives(index, queries)
