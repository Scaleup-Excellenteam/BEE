"""Tests for the semantic index builder.

The builder is always given a fake embedder, so these tests never touch
the Gemini API.
"""

from unittest.mock import Mock

import pytest

from src.models import SentenceRecord
from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import CacheError, save_embeddings
from src.semantic.index_builder import (
    build_embeddings,
    load_or_build_embeddings,
    open_or_build_store,
)


def fake_embedder(text: str) -> list[float]:
    """Return a deterministic vector that depends on the text.

    The values are whole numbers so they survive float32 storage exactly,
    which keeps these tests free of approximate comparisons.
    """
    return [1.0, 2.0, float(len(text))]


def sample_records():
    """Return two corpus records with distinct sources and offsets."""
    return [
        SentenceRecord(
            original_sentence="Authentication is required.",
            normalized_sentence="authentication is required",
            source_text="sg244986.txt",
            offset=125,
        ),
        SentenceRecord(
            original_sentence="The server returned an error.",
            normalized_sentence="the server returned an error",
            source_text="other/source.txt",
            offset=7,
        ),
    ]


def test_cache_miss_builds_embeddings_from_records(tmp_path):
    path = str(tmp_path / "cache")

    items = load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=fake_embedder,
    )

    assert items == [
        EmbeddedSentence(
            sentence="Authentication is required.",
            source_text="sg244986.txt",
            offset=125,
            embedding=[1.0, 2.0, 27.0],
        ),
        EmbeddedSentence(
            sentence="The server returned an error.",
            source_text="other/source.txt",
            offset=7,
            embedding=[1.0, 2.0, 29.0],
        ),
    ]


def test_cache_miss_writes_the_cache(tmp_path):
    path = str(tmp_path / "cache")

    load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=fake_embedder,
    )

    assert (tmp_path / "cache" / "manifest.json").is_file()


def test_cache_hit_does_not_call_the_embedder(tmp_path):
    path = str(tmp_path / "cache")
    embedder = Mock(side_effect=fake_embedder)

    first = load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=embedder,
    )
    calls_after_build = embedder.call_count

    second = load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=embedder,
    )

    assert calls_after_build == 2
    assert embedder.call_count == 2
    assert second == first


def test_cache_hit_ignores_the_records_it_is_given(tmp_path):
    path = str(tmp_path / "cache")
    load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=fake_embedder,
    )
    embedder = Mock()

    items = load_or_build_embeddings([], cache_path=path, embedder=embedder)

    assert [item.sentence for item in items] == [
        "Authentication is required.",
        "The server returned an error.",
    ]
    embedder.assert_not_called()


def test_embeds_the_original_sentence_not_the_normalized_one(tmp_path):
    embedder = Mock(side_effect=fake_embedder)

    load_or_build_embeddings(
        sample_records(),
        cache_path=str(tmp_path / "cache"),
        embedder=embedder,
    )

    embedded_texts = [call.args[0] for call in embedder.call_args_list]
    assert embedded_texts == [
        "Authentication is required.",
        "The server returned an error.",
    ]


def test_preserves_source_text_and_offset_from_the_records(tmp_path):
    items = load_or_build_embeddings(
        sample_records(),
        cache_path=str(tmp_path / "cache"),
        embedder=fake_embedder,
    )

    assert [(item.source_text, item.offset) for item in items] == [
        ("sg244986.txt", 125),
        ("other/source.txt", 7),
    ]


def test_survives_a_full_save_and_reload_cycle(tmp_path):
    path = str(tmp_path / "cache")
    built = load_or_build_embeddings(
        sample_records(),
        cache_path=path,
        embedder=fake_embedder,
    )

    reloaded = load_or_build_embeddings([], cache_path=path)

    assert reloaded == built


def test_handles_an_empty_record_list(tmp_path):
    items = load_or_build_embeddings(
        [],
        cache_path=str(tmp_path / "cache"),
        embedder=fake_embedder,
    )

    assert items == []


def test_corrupted_cache_raises_instead_of_rebuilding(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(
        [
            EmbeddedSentence(
                sentence="cached",
                source_text="a.txt",
                offset=1,
                embedding=[0.5],
            )
        ],
        path,
    )
    (tmp_path / "cache" / "manifest.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    embedder = Mock()

    with pytest.raises(CacheError):
        load_or_build_embeddings(
            sample_records(),
            cache_path=path,
            embedder=embedder,
        )

    embedder.assert_not_called()


def test_build_embeddings_can_be_used_on_its_own(tmp_path):
    items = build_embeddings(sample_records(), fake_embedder)

    assert len(items) == 2
    assert items[0].embedding == [1.0, 2.0, 27.0]


def fake_batch_embedder(texts):
    """Embed a whole batch at once, mirroring embed_texts."""
    return [fake_embedder(text) for text in texts]


def test_store_build_embeds_every_record(tmp_path):
    path = str(tmp_path / "store")

    store = open_or_build_store(
        sample_records(),
        cache_path=path,
        batch_embedder=fake_batch_embedder,
    )

    with store:
        assert [item.sentence for item in store] == [
            "Authentication is required.",
            "The server returned an error.",
        ]
        assert [item.source_text for item in store] == [
            "sg244986.txt",
            "other/source.txt",
        ]
        assert [item.offset for item in store] == [125, 7]
        assert [item.embedding for item in store] == [
            [1.0, 2.0, 27.0],
            [1.0, 2.0, 29.0],
        ]


def test_store_build_sends_batches_not_single_texts(tmp_path):
    records = [
        SentenceRecord(
            original_sentence=f"sentence number {index}",
            normalized_sentence=f"sentence number {index}",
            source_text="a.txt",
            offset=index,
        )
        for index in range(5)
    ]
    batch_embedder = Mock(side_effect=fake_batch_embedder)

    store = open_or_build_store(
        records,
        cache_path=str(tmp_path / "store"),
        batch_embedder=batch_embedder,
        batch_size=2,
    )
    store.close()

    assert batch_embedder.call_count == 3
    assert [len(call.args[0]) for call in batch_embedder.call_args_list] == [
        2,
        2,
        1,
    ]


def test_store_build_embeds_the_original_sentence(tmp_path):
    batch_embedder = Mock(side_effect=fake_batch_embedder)

    store = open_or_build_store(
        sample_records(),
        cache_path=str(tmp_path / "store"),
        batch_embedder=batch_embedder,
    )
    store.close()

    assert batch_embedder.call_args_list[0].args[0] == [
        "Authentication is required.",
        "The server returned an error.",
    ]


def test_finished_store_is_not_rebuilt(tmp_path):
    path = str(tmp_path / "store")
    first = open_or_build_store(
        sample_records(),
        cache_path=path,
        batch_embedder=fake_batch_embedder,
    )
    first.close()

    batch_embedder = Mock()
    second = open_or_build_store(
        sample_records(),
        cache_path=path,
        batch_embedder=batch_embedder,
    )

    with second:
        assert len(second) == 2

    batch_embedder.assert_not_called()


def test_interrupted_store_build_resumes_without_re_embedding(tmp_path):
    path = str(tmp_path / "store")
    records = [
        SentenceRecord(
            original_sentence=f"sentence number {index}",
            normalized_sentence=f"sentence number {index}",
            source_text="a.txt",
            offset=index,
        )
        for index in range(4)
    ]

    def fail_after_two_batches(texts):
        if fail_after_two_batches.calls == 2:
            raise RuntimeError("simulated crash")
        fail_after_two_batches.calls += 1
        return fake_batch_embedder(texts)

    fail_after_two_batches.calls = 0

    with pytest.raises(RuntimeError, match="simulated crash"):
        open_or_build_store(
            records,
            cache_path=path,
            batch_embedder=fail_after_two_batches,
            batch_size=1,
        )

    resumed_embedder = Mock(side_effect=fake_batch_embedder)
    store = open_or_build_store(
        records,
        cache_path=path,
        batch_embedder=resumed_embedder,
        batch_size=1,
    )

    with store:
        assert len(store) == 4
        assert [item.offset for item in store] == [0, 1, 2, 3]

    embedded_after_resume = [
        call.args[0][0] for call in resumed_embedder.call_args_list
    ]
    assert embedded_after_resume == ["sentence number 2", "sentence number 3"]


def test_store_build_reports_progress(tmp_path):
    seen = []

    store = open_or_build_store(
        sample_records(),
        cache_path=str(tmp_path / "store"),
        batch_embedder=fake_batch_embedder,
        batch_size=1,
        progress=seen.append,
    )
    store.close()

    assert seen == [1, 2]


def test_store_build_handles_no_records(tmp_path):
    store = open_or_build_store(
        [],
        cache_path=str(tmp_path / "store"),
        batch_embedder=fake_batch_embedder,
    )

    with store:
        assert len(store) == 0
        assert list(store) == []


def test_invalid_batch_size_raises(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        open_or_build_store(
            sample_records(),
            cache_path=str(tmp_path / "store"),
            batch_embedder=fake_batch_embedder,
            batch_size=0,
        )
