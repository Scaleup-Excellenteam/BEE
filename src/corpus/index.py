"""Character-trigram candidate index.

This module only performs CANDIDATE RETRIEVAL.

The contract with the matching stage is deliberately one-sided:

    False positives are allowed.
    False negatives are forbidden.

``get_candidates`` may return sentences that turn out not to match, but it
must never drop a sentence that could legally match the query with an exact
substring match, one substitution, one missing character or one extra
character.  The final decision is made downstream, not here.


Why character trigrams
----------------------

Every string is cut into overlapping windows of three characters:

    "hello"  ->  "hel", "ell", "llo"

The cut is purely character based.  Spaces are ordinary characters, so
"hello world" also produces the boundary trigrams "o w", " wo", ... .  That
is what lets a query start in the middle of a word ("puter sci").


Why the threshold is  U - 3
---------------------------

Let ``q`` be the normalized query of length ``m``, and let a sentence ``s``
match when some substring ``t`` of ``s`` is within one edit of ``q``.
The query has trigram positions ``i = 0 .. m-3``.

A single edit damages at most three of those positions:

  * exact match: nothing is damaged.
  * substitution at index j: position i is unharmed unless it covers j, so
    only i in {j-2, j-1, j} is damaged  ->  at most 3.
  * a character missing from the query (insert c into q at j, |t| = m+1):
    positions with i+2 < j still read t[i:i+3], positions with i >= j still
    read t[i+1:i+4], so only i in {j-2, j-1} is damaged  ->  at most 2.
  * an extra character in the query (delete q[j], |t| = m-1): positions with
    i+2 < j still read t[i:i+3], positions with i > j still read
    t[i-1:i+2], so only i in {j-2, j-1, j} is damaged  ->  at most 3.

Each position carries exactly one distinct trigram, and a distinct trigram
only disappears if every one of its positions is damaged.  Damaging at most
three positions therefore kills at most three DISTINCT trigrams, so at least
``U - 3`` distinct query trigrams still occur inside ``t``.  Since ``t`` is a
substring of ``s``, every trigram of ``t`` is also a trigram of ``s``, and
counting over the whole sentence can only raise the count.  Hence:

    a legally matching sentence always shares at least  U - 3
    distinct trigrams with the query.

Two consequences:

  * ``U <= 3`` makes the threshold zero or negative, so every sentence
    qualifies and returning the whole corpus is the correct answer (this also
    covers queries shorter than three characters, which have no trigrams).
  * if fewer than ``U - 3`` query trigrams occur anywhere in the corpus, no
    sentence can reach the threshold and the empty list is safe.


Memory
------

Posting lists are stored as ``array('I')`` instead of ``set[int]``.  A set of
Python ints costs roughly 100 bytes per entry (a heap-allocated int object
plus a sparsely filled hash table slot); an ``array('I')`` costs 4 bytes per
entry in one contiguous buffer.  On a corpus of a few million sentences that
is the difference between gigabytes and hundreds of megabytes.

Records are never copied into the postings.  A posting entry is an integer
sentence id that indexes ``self.records``.
"""

import sys
import time
from array import array
from collections import defaultdict

from src.models import SentenceRecord

# Size of a character n-gram.
TRIGRAM_SIZE = 3

# An edit can destroy at most TRIGRAM_SIZE trigram positions of the query,
# which is exactly the slack the threshold has to allow for.
MAX_DAMAGED_TRIGRAMS = TRIGRAM_SIZE

# Type code of the posting arrays: C unsigned int, at least 4 bytes.
POSTING_TYPE_CODE = "I"


def _iter_trigrams(text: str):
    """Yield every character trigram of ``text``, position by position."""
    for start in range(len(text) - TRIGRAM_SIZE + 1):
        yield text[start:start + TRIGRAM_SIZE]


def _distinct_trigrams(text: str) -> set[str]:
    """Return the distinct character trigrams of ``text``.

    A trigram that repeats inside one string is collapsed to a single entry,
    so a sentence is posted to each of its trigrams exactly once.
    """
    return set(_iter_trigrams(text))


class CorpusIndex:
    """Inverted character-trigram index over a list of sentence records.

    Built once during offline initialization and reused for every query.
    """

    def __init__(self, records: list[SentenceRecord]):
        self.records = records
        self._postings: dict[str, array] = {}
        self._total_postings = 0
        self._build_seconds = 0.0

        self._build()

    def _build(self) -> None:
        """Fill the trigram index from ``self.records``.

        Sentences are visited in increasing id order and each sentence is
        appended at most once per trigram, so every posting list ends up
        sorted and free of duplicates without any extra work.
        """
        start_time = time.perf_counter()

        postings: dict[str, array] = defaultdict(
            lambda: array(POSTING_TYPE_CODE)
        )
        total_postings = 0

        for sentence_id, record in enumerate(self.records):
            for trigram in _distinct_trigrams(record.normalized_sentence):
                postings[trigram].append(sentence_id)
                total_postings += 1

        self._postings = dict(postings)
        self._total_postings = total_postings
        self._build_seconds = time.perf_counter() - start_time

    def get_candidates(self, query: str) -> list[SentenceRecord]:
        """Return every record that could still match the normalized query.

        The result is a superset of the real matches: the caller is expected
        to run the final edit check on each candidate.
        """
        # Team decision: an empty query retrieves nothing.
        if not query:
            return []

        query_trigrams = _distinct_trigrams(query)
        unique_count = len(query_trigrams)

        # Too few trigrams for the count to prove anything, so no sentence
        # may be discarded.
        if unique_count <= MAX_DAMAGED_TRIGRAMS:
            return list(self.records)

        required_shared = unique_count - MAX_DAMAGED_TRIGRAMS

        posting_lists = [
            self._postings[trigram]
            for trigram in query_trigrams
            if trigram in self._postings
        ]

        # Even a sentence holding all of them could not reach the threshold,
        # so nothing in the corpus can match.
        if len(posting_lists) < required_shared:
            return []

        shared_counts: dict[int, int] = defaultdict(int)
        for posting_list in posting_lists:
            for sentence_id in posting_list:
                shared_counts[sentence_id] += 1

        # Keys of a dict are unique, so one sentence id yields one record.
        return [
            self.records[sentence_id]
            for sentence_id in sorted(shared_counts)
            if shared_counts[sentence_id] >= required_shared
        ]

    def stats(self) -> dict:
        """Return cheap structural statistics about the built index."""
        keys_bytes = sum(sys.getsizeof(key) for key in self._postings)
        buffers_bytes = sum(
            posting_list.buffer_info()[1] * posting_list.itemsize
            for posting_list in self._postings.values()
        )

        return {
            "record_count": len(self.records),
            "distinct_trigrams": len(self._postings),
            "total_postings": self._total_postings,
            "build_seconds": self._build_seconds,
            "approximate_index_bytes": (
                sys.getsizeof(self._postings) + keys_bytes + buffers_bytes
            ),
        }
