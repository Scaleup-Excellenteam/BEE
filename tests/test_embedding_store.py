"""Tests for memory mapped, resumable embedding storage.

Every test uses tiny synthetic vectors and never touches the Gemini API.
Stores are always closed, because an open memory map keeps the file
locked on Windows.
"""

import numpy as np
import pytest

from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import (
    CacheError,
    cache_files,
    load_embeddings,
    read_manifest,
    save_embeddings,
)
from src.semantic.embedding_store import (
    EmbeddingStore,
    EmbeddingStoreWriter,
)


def sample_items():
    """Return three embedded sentences with whole number vectors.

    Whole numbers survive float32 storage exactly, which keeps these
    tests free of approximate comparisons.
    """
    return [
        EmbeddedSentence(
            sentence="Authentication is required.",
            source_text="sg244986.txt",
            offset=125,
            embedding=[1.0, 2.0, 3.0],
        ),
        EmbeddedSentence(
            sentence="The server returned an error.",
            source_text="other/source.txt",
            offset=7,
            embedding=[4.0, 5.0, 6.0],
        ),
        EmbeddedSentence(
            sentence="Grüße, mundo — ok",
            source_text="unicode.txt",
            offset=3,
            embedding=[7.0, 8.0, 9.0],
        ),
    ]


@pytest.fixture
def saved_cache(tmp_path):
    """Return the path of a cache written by the small sample writer."""
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)

    return path


def test_store_reads_a_cache_written_by_save_embeddings(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        assert len(store) == 3
        assert store.dim == 3
        assert store.is_complete


def test_vectors_are_a_memory_map_not_a_python_list(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        vectors = store.vectors()

        assert isinstance(vectors, np.memmap)
        assert vectors.shape == (3, 3)
        assert vectors.dtype == np.float32


def test_iterating_yields_every_record_in_order(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        items = list(store)

    assert items == sample_items()


def test_iterating_preserves_non_ascii_sentences(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        sentences = [item.sentence for item in store]

    assert sentences[2] == "Grüße, mundo — ok"


def test_get_returns_one_full_record(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        assert store.get(1) == sample_items()[1]


def test_get_reaches_the_last_record(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        assert store.get(2) == sample_items()[2]


def test_metadata_returns_the_stored_fields(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        assert store.metadata(0) == {
            "sentence": "Authentication is required.",
            "source_text": "sg244986.txt",
            "offset": 125,
        }


def test_out_of_range_index_raises(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        with pytest.raises(IndexError):
            store.get(3)

        with pytest.raises(IndexError):
            store.get(-1)


def test_blocks_cover_every_vector_exactly_once(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        blocks = [(start, block.copy()) for start, block in
                  store.iter_blocks(block_size=2)]

    assert [start for start, _ in blocks] == [0, 2]
    assert [block.shape for _, block in blocks] == [(2, 3), (1, 3)]

    stacked = np.vstack([block for _, block in blocks])
    assert stacked.tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_a_single_block_holds_everything_by_default(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        blocks = list(store.iter_blocks())

    assert len(blocks) == 1
    assert blocks[0][0] == 0


def test_invalid_block_size_raises(saved_cache):
    with EmbeddingStore(saved_cache) as store:
        with pytest.raises(ValueError, match="at least 1"):
            list(store.iter_blocks(block_size=0))


def test_empty_store_is_usable(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings([], path)

    with EmbeddingStore(path) as store:
        assert len(store) == 0
        assert list(store) == []
        assert store.vectors().shape == (0, 0)


def test_store_rejects_a_vector_file_that_does_not_match(saved_cache):
    vectors_path = cache_files(saved_cache)[3]
    vectors_path.write_bytes(vectors_path.read_bytes()[:-4])

    with pytest.raises(CacheError, match="vector file size"):
        EmbeddingStore(saved_cache)


def test_store_rejects_an_offset_file_that_does_not_match(saved_cache):
    offsets_path = cache_files(saved_cache)[2]
    offsets_path.write_bytes(offsets_path.read_bytes()[:-8])

    with pytest.raises(CacheError, match="offset file size"):
        EmbeddingStore(saved_cache)


def test_writer_builds_a_store_readable_by_both_readers(tmp_path):
    path = str(tmp_path / "cache")

    with EmbeddingStoreWriter(path, dim=3) as writer:
        writer.append(sample_items()[:2])
        writer.append(sample_items()[2:])
        writer.finish()

    with EmbeddingStore(path) as store:
        assert list(store) == sample_items()

    assert load_embeddings(path) == sample_items()


def test_writer_appends_in_batches_without_rewriting(tmp_path):
    path = str(tmp_path / "cache")

    with EmbeddingStoreWriter(path, dim=3) as writer:
        for item in sample_items():
            writer.append([item])
        writer.finish()

    with EmbeddingStore(path) as store:
        assert [item.offset for item in store] == [125, 7, 3]


def test_an_unfinished_build_is_marked_incomplete(tmp_path):
    path = str(tmp_path / "cache")

    writer = EmbeddingStoreWriter(path, dim=3)
    writer.append(sample_items()[:1])
    writer.close()

    assert read_manifest(path)["complete"] is False

    with EmbeddingStore(path) as store:
        assert store.is_complete is False
        assert len(store) == 1


def test_eager_loading_refuses_an_unfinished_build(tmp_path):
    path = str(tmp_path / "cache")

    writer = EmbeddingStoreWriter(path, dim=3)
    writer.append(sample_items()[:1])
    writer.close()

    with pytest.raises(CacheError, match="unfinished build"):
        load_embeddings(path)


def test_an_interrupted_build_resumes_where_it_stopped(tmp_path):
    path = str(tmp_path / "cache")

    writer = EmbeddingStoreWriter(path, dim=3)
    writer.append(sample_items()[:1])
    writer.close()

    resumed = EmbeddingStoreWriter(path, dim=3)
    assert resumed.count == 1

    resumed.append(sample_items()[1:])
    resumed.finish()

    with EmbeddingStore(path) as store:
        assert list(store) == sample_items()


def test_resuming_discards_a_half_written_batch(tmp_path):
    path = str(tmp_path / "cache")

    writer = EmbeddingStoreWriter(path, dim=3)
    writer.append(sample_items()[:1])
    writer.close()

    # Simulate a crash after vectors were written but before the manifest
    # was updated: extra bytes the manifest does not know about.
    vectors_path = cache_files(path)[3]
    vectors_path.write_bytes(vectors_path.read_bytes() + b"\x00" * 12)

    resumed = EmbeddingStoreWriter(path, dim=3)
    assert resumed.count == 1

    resumed.append(sample_items()[1:])
    resumed.finish()

    with EmbeddingStore(path) as store:
        assert list(store) == sample_items()


def test_writer_rejects_a_wrong_sized_embedding(tmp_path):
    path = str(tmp_path / "cache")

    with EmbeddingStoreWriter(path, dim=3) as writer:
        with pytest.raises(CacheError, match="Expected 3 dimensional"):
            writer.append(
                [
                    EmbeddedSentence(
                        sentence="s",
                        source_text="a.txt",
                        offset=1,
                        embedding=[1.0, 2.0],
                    )
                ]
            )


def test_writer_refuses_to_append_a_different_dimension(tmp_path):
    path = str(tmp_path / "cache")

    with EmbeddingStoreWriter(path, dim=3) as writer:
        writer.append(sample_items()[:1])
        writer.finish()

    with pytest.raises(CacheError, match="cannot append"):
        EmbeddingStoreWriter(path, dim=8)


def test_appending_nothing_changes_nothing(tmp_path):
    path = str(tmp_path / "cache")

    with EmbeddingStoreWriter(path, dim=3) as writer:
        writer.append(sample_items()[:1])
        writer.append([])
        writer.finish()

    with EmbeddingStore(path) as store:
        assert len(store) == 1
