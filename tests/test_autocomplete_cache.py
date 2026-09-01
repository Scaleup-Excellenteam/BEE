"""Integration tests for caching normalized autocomplete queries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from unittest.mock import Mock

import autocomplete
from src.models import AutoCompleteData, SentenceRecord


def _record(
    sentence: str,
    *,
    source_text: str = "source.txt",
    offset: int = 1,
) -> SentenceRecord:
    return SentenceRecord(
        original_sentence=sentence,
        normalized_sentence=sentence.lower(),
        source_text=source_text,
        offset=offset,
    )


def _index_with_result(sentence: str = "Hello world") -> Mock:
    index = Mock()
    index.get_candidates.return_value = [_record(sentence)]
    return index


def test_normal_query_cache_miss_then_hit() -> None:
    index = _index_with_result()
    autocomplete.set_corpus_index(index)

    first = autocomplete.get_best_k_completions("hello")
    second = autocomplete.get_best_k_completions("hello")

    assert first == second
    index.get_candidates.assert_called_once_with("hello")
    info = autocomplete.get_query_cache_info()
    assert (info.hits, info.misses) == (1, 1)


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
        _record(f"Hello {number}", offset=number)
        for number in range(7)
    ]
    autocomplete.set_corpus_index(index)

    first = autocomplete.get_best_k_completions("hello")
    second = autocomplete.get_best_k_completions("HELLO!")

    assert first == second
    assert len(first) == 5
    index.get_candidates.assert_called_once_with("hello")


def test_empty_autocomplete_results_are_cached() -> None:
    index = Mock()
    index.get_candidates.return_value = []
    autocomplete.set_corpus_index(index)

    assert autocomplete.get_best_k_completions("missing") == []
    assert autocomplete.get_best_k_completions("MISSING!") == []

    index.get_candidates.assert_called_once_with("missing")
    info = autocomplete.get_query_cache_info()
    assert (info.hits, info.misses, info.size) == (1, 1, 1)


def test_cached_results_preserve_order_metadata_and_defensive_copies() -> None:
    index = Mock()
    index.get_candidates.return_value = [
        _record("Zulu hello", source_text="z.txt", offset=30),
        _record("Alpha hello", source_text="a.txt", offset=10),
        _record("Beta hello", source_text="b.txt", offset=20),
    ]
    autocomplete.set_corpus_index(index)

    miss_results = autocomplete.get_best_k_completions("hello")
    hit_results = autocomplete.get_best_k_completions("HELLO!")

    expected = [
        AutoCompleteData("Alpha hello", "a.txt", 10, 10),
        AutoCompleteData("Beta hello", "b.txt", 20, 10),
        AutoCompleteData("Zulu hello", "z.txt", 30, 10),
    ]
    assert miss_results == hit_results == expected
    assert miss_results is not hit_results
    assert all(
        miss_result is not hit_result
        for miss_result, hit_result in zip(miss_results, hit_results)
    )

    miss_results[0].completed_sentence = "Mutated"
    assert autocomplete.get_best_k_completions("hello") == expected


def test_request_cannot_publish_old_index_results_into_new_index_cache() -> None:
    entered_old_index = Event()
    release_old_index = Event()

    class BlockingIndex:
        def get_candidates(self, query: str):
            entered_old_index.set()
            assert release_old_index.wait(timeout=5)
            return [_record("Old hello", source_text="old.txt")]

    old_index = BlockingIndex()
    new_index = _index_with_result("New hello")
    autocomplete.set_corpus_index(old_index)

    with ThreadPoolExecutor(max_workers=1) as executor:
        old_request = executor.submit(
            autocomplete.get_best_k_completions,
            "hello",
        )
        assert entered_old_index.wait(timeout=5)
        autocomplete.set_corpus_index(new_index)
        release_old_index.set()
        old_results = old_request.result(timeout=5)

    new_results = autocomplete.get_best_k_completions("hello")

    assert [result.completed_sentence for result in old_results] == [
        "Old hello"
    ]
    assert [result.completed_sentence for result in new_results] == [
        "New hello"
    ]
    new_index.get_candidates.assert_called_once_with("hello")


def test_concurrent_duplicate_misses_do_not_increase_lfu_frequency() -> None:
    duplicate_barrier = Barrier(2)
    call_lock = Lock()
    shared_calls = 0

    class ConcurrentIndex:
        def get_candidates(self, query: str):
            nonlocal shared_calls
            if query == "shared":
                with call_lock:
                    shared_calls += 1
                    current_call = shared_calls
                if current_call <= 2:
                    duplicate_barrier.wait(timeout=5)
            return [_record(f"{query} result")]

    autocomplete.set_corpus_index(
        ConcurrentIndex(),
        query_cache_capacity=2,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                autocomplete.get_best_k_completions,
                "shared",
            )
            for _ in range(2)
        ]
        assert [future.result(timeout=5) for future in futures]

    info = autocomplete.get_query_cache_info()
    assert (info.hits, info.misses, info.size) == (0, 2, 1)

    autocomplete.get_best_k_completions("other")
    autocomplete.get_best_k_completions("third")
    autocomplete.get_best_k_completions("shared")

    assert shared_calls == 3
