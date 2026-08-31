from src.corpus.loader import load_corpus
from src.models import SentenceRecord


def test_loads_records_with_all_fields(tmp_path):
    (tmp_path / "a.txt").write_text("Hello World\n", encoding="utf-8")

    records = load_corpus(str(tmp_path))

    assert records == [
        SentenceRecord(
            original_sentence="Hello World",
            normalized_sentence="hello world",
            source_text="a.txt",
            offset=1,
        )
    ]


def test_offset_is_one_based_line_number(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    records = load_corpus(str(tmp_path))

    assert [record.offset for record in records] == [1, 2, 3]


def test_blank_lines_are_skipped_but_still_count_offsets(tmp_path):
    (tmp_path / "a.txt").write_text("one\n\n   \nfour\n", encoding="utf-8")

    records = load_corpus(str(tmp_path))

    assert [(r.original_sentence, r.offset) for r in records] == [
        ("one", 1),
        ("four", 4),
    ]


def test_source_text_uses_posix_separators(tmp_path):
    nested = tmp_path / "docs" / "whatsnew"
    nested.mkdir(parents=True)
    (nested / "index.txt").write_text("line\n", encoding="utf-8")

    records = load_corpus(str(tmp_path))

    assert records[0].source_text == "docs/whatsnew/index.txt"
    assert "\\" not in records[0].source_text


def test_reads_nested_files_recursively(tmp_path):
    (tmp_path / "top.txt").write_text("top\n", encoding="utf-8")
    nested = tmp_path / "deep"
    nested.mkdir()
    (nested / "inner.txt").write_text("inner\n", encoding="utf-8")

    sources = {record.source_text for record in load_corpus(str(tmp_path))}

    assert sources == {"top.txt", "deep/inner.txt"}


def test_ignores_non_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("kept\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("dropped\n", encoding="utf-8")

    records = load_corpus(str(tmp_path))

    assert [record.original_sentence for record in records] == ["kept"]


def test_empty_directory_yields_no_records(tmp_path):
    assert load_corpus(str(tmp_path)) == []


def test_carriage_returns_are_stripped(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hello\r\nworld\r\n")

    records = load_corpus(str(tmp_path))

    assert [record.original_sentence for record in records] == [
        "hello",
        "world",
    ]
