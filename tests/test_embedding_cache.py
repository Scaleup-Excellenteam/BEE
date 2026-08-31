"""Tests for the persistent embedding cache."""

import json

import pytest

from src.semantic import embedding_cache
from src.semantic.contracts import EmbeddedSentence
from src.semantic.embedding_cache import (
    CacheError,
    load_embeddings,
    save_embeddings,
)


def sample_items():
    """Return two embedded sentences with distinct field values."""
    return [
        EmbeddedSentence(
            sentence="Authentication is required.",
            source_text="sg244986.txt",
            offset=125,
            embedding=[0.12, -0.43, 0.81],
        ),
        EmbeddedSentence(
            sentence="The server returned an error.",
            source_text="other/source.txt",
            offset=7,
            embedding=[1.5, 0.0, -2.25],
        ),
    ]


def test_saves_a_cache_directory_with_three_files(tmp_path):
    path = str(tmp_path / "cache")

    save_embeddings(sample_items(), path)

    assert embedding_cache.cache_exists(path)
    assert (tmp_path / "cache" / "manifest.json").is_file()
    assert (tmp_path / "cache" / "metadata.jsonl").is_file()
    assert (tmp_path / "cache" / "vectors.f32").is_file()


def test_round_trip_preserves_metadata_exactly(tmp_path):
    path = str(tmp_path / "cache")
    items = sample_items()

    save_embeddings(items, path)
    loaded = load_embeddings(path)

    assert [
        (item.sentence, item.source_text, item.offset) for item in loaded
    ] == [(item.sentence, item.source_text, item.offset) for item in items]


def test_round_trip_preserves_vectors_within_float32_precision(tmp_path):
    path = str(tmp_path / "cache")
    items = sample_items()

    save_embeddings(items, path)
    loaded = load_embeddings(path)

    for saved, reloaded in zip(items, loaded):
        assert reloaded.embedding == pytest.approx(saved.embedding, rel=1e-6)


def test_float32_storage_is_lossy_by_design(tmp_path):
    """Document the tradeoff: close enough for cosine, not identical."""
    path = str(tmp_path / "cache")
    save_embeddings(
        [
            EmbeddedSentence(
                sentence="s",
                source_text="a.txt",
                offset=1,
                embedding=[0.12],
            )
        ],
        path,
    )

    stored = load_embeddings(path)[0].embedding[0]

    assert stored != 0.12
    assert stored == pytest.approx(0.12, rel=1e-6)


def test_round_trip_preserves_each_field_individually(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)

    loaded = load_embeddings(path)

    assert loaded[0].sentence == "Authentication is required."
    assert loaded[0].source_text == "sg244986.txt"
    assert loaded[0].offset == 125
    assert loaded[0].embedding == pytest.approx([0.12, -0.43, 0.81], rel=1e-6)


def test_round_trip_preserves_order_of_multiple_records(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)

    loaded = load_embeddings(path)

    assert [item.sentence for item in loaded] == [
        "Authentication is required.",
        "The server returned an error.",
    ]
    assert [item.offset for item in loaded] == [125, 7]


def test_round_trip_handles_an_empty_list(tmp_path):
    path = str(tmp_path / "cache")

    save_embeddings([], path)

    assert load_embeddings(path) == []


def test_round_trip_preserves_non_ascii_sentences(tmp_path):
    path = str(tmp_path / "cache")
    items = [
        EmbeddedSentence(
            sentence="Grüße, mundo — ok",
            source_text="unicode.txt",
            offset=3,
            embedding=[0.5],
        )
    ]

    save_embeddings(items, path)
    loaded = load_embeddings(path)

    assert loaded[0].sentence == "Grüße, mundo — ok"
    assert loaded[0].embedding == pytest.approx([0.5])


def test_saving_ragged_embeddings_raises(tmp_path):
    items = sample_items()
    items[1].embedding = [1.0, 2.0]

    with pytest.raises(CacheError, match="same length"):
        save_embeddings(items, str(tmp_path / "cache"))


def test_loading_a_missing_cache_raises(tmp_path):
    with pytest.raises(CacheError, match="No embedding cache found"):
        load_embeddings(str(tmp_path / "absent"))


def test_corrupted_manifest_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    (tmp_path / "cache" / "manifest.json").write_text(
        "{not json",
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="Unreadable cache manifest"):
        load_embeddings(path)


def test_unsupported_cache_version_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    (tmp_path / "cache" / "manifest.json").write_text(
        json.dumps({"version": 99, "count": 2, "dim": 3}),
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="Unsupported cache version"):
        load_embeddings(path)


def test_corrupted_metadata_line_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    (tmp_path / "cache" / "metadata.jsonl").write_text(
        '{"sentence": "ok", "source_text": "a.txt", "offset": 1}\n{broken\n',
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="Corrupted cache metadata on line 2"):
        load_embeddings(path)


def test_metadata_missing_a_field_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    (tmp_path / "cache" / "metadata.jsonl").write_text(
        '{"sentence": "ok", "offset": 1}\n',
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="missing source_text"):
        load_embeddings(path)


def test_metadata_count_mismatch_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    (tmp_path / "cache" / "metadata.jsonl").write_text(
        '{"sentence": "ok", "source_text": "a.txt", "offset": 1}\n',
        encoding="utf-8",
    )

    with pytest.raises(CacheError, match="promises 2 sentences"):
        load_embeddings(path)


def test_truncated_vector_file_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    vectors_path = tmp_path / "cache" / "vectors.f32"
    vectors_path.write_bytes(vectors_path.read_bytes()[:-16])

    with pytest.raises(CacheError, match="truncated"):
        load_embeddings(path)


def test_oversized_vector_file_raises(tmp_path):
    path = str(tmp_path / "cache")
    save_embeddings(sample_items(), path)
    vectors_path = tmp_path / "cache" / "vectors.f32"
    vectors_path.write_bytes(vectors_path.read_bytes() + b"\x00" * 8)

    with pytest.raises(CacheError, match="does not match the manifest"):
        load_embeddings(path)
