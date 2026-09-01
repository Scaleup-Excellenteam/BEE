"""Integration tests across the corpus, matching, and autocomplete stages."""

import autocomplete
from src.corpus.index import CorpusIndex
from src.corpus.normalizer import normalize_text
from src.models import AutoCompleteData, SentenceRecord


def test_real_autocomplete_integration():
    autocomplete.set_corpus_index(None)

    records = [
        SentenceRecord(
            original_sentence="Hello World",
            normalized_sentence=normalize_text("Hello World"),
            source_text="greetings.txt",
            offset=1,
        ),
        SentenceRecord(
            original_sentence="Unrelated sentence",
            normalized_sentence=normalize_text("Unrelated sentence"),
            source_text="other.txt",
            offset=2,
        ),
    ]
    index = CorpusIndex(records)
    autocomplete.set_corpus_index(index)

    results = autocomplete.get_best_k_completions("HELLO!")

    assert results == [
        AutoCompleteData(
            completed_sentence="Hello World",
            source_text="greetings.txt",
            offset=1,
            score=10,
        )
    ]
