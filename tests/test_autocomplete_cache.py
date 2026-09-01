"""Integration tests for caching normalized autocomplete queries."""

from unittest.mock import Mock

import autocomplete
from src.models import SentenceRecord


def _index_with_result(sentence: str = "Hello world") -> Mock:
    index = Mock()
    index.get_candidates.return_value = [
        SentenceRecord(
            original_sentence=sentence,
            normalized_sentence=sentence.lower(),
            source_text="source.txt",
            offset=1,
        )
    ]
    return index


def test_cache_reuses_normalized_query_results() -> None:
    index = _index_with_result()
    autocomplete.set_corpus_index(index, query_cache_capacity=2)

    first = autocomplete.get_best_k_completions("HELLO!!!")
    second = autocomplete.get_best_k_completions(" hello ")

    assert first == second
    index.get_candidates.assert_called_once_with("hello")
    info = autocomplete.get_query_cache_info()
    assert info.capacity == 2
    assert info.size == 1
    assert info.hits == 1
    assert info.misses == 1


def test_setting_new_corpus_index_clears_cached_results() -> None:
    first_index = _index_with_result("First hello")
    autocomplete.set_corpus_index(first_index)
    autocomplete.get_best_k_completions("hello")
    autocomplete.get_best_k_completions("hello")
    first_index.get_candidates.assert_called_once_with("hello")

    second_index = _index_with_result("Second hello")
    autocomplete.set_corpus_index(second_index)
    results = autocomplete.get_best_k_completions("hello")

    second_index.get_candidates.assert_called_once_with("hello")
    assert [result.completed_sentence for result in results] == [
        "Second hello"
    ]
    info = autocomplete.get_query_cache_info()
    assert info.hits == 0
    assert info.misses == 1


def test_exact_match_fast_path_is_cached() -> None:
    index = Mock()
    index.get_candidates.return_value = [
        SentenceRecord(
            original_sentence=f"Hello {number}",
            normalized_sentence=f"hello {number}",
            source_text="source.txt",
            offset=number,
        )
        for number in range(5)
    ]
    autocomplete.set_corpus_index(index)

    first = autocomplete.get_best_k_completions("hello")
    second = autocomplete.get_best_k_completions("HELLO!")

    assert first == second
    index.get_candidates.assert_called_once_with("hello")
