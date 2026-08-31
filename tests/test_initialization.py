from pathlib import Path
from zipfile import ZipFile

from src.corpus.index import CorpusIndex
from src.corpus.initialization import initialize_corpus


def test_initialize_corpus_returns_a_ready_index(tmp_path):
    archive_path = tmp_path / "small.zip"
    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("docs/a.txt", "Computer Science\n")
        zip_file.writestr("docs/b.txt", "Hello World\n")

    index = initialize_corpus(
        str(archive_path),
        str(tmp_path / "extracted"),
    )

    assert isinstance(index, CorpusIndex)
    assert len(index.records) == 2

    candidates = index.get_candidates("puter sci")
    assert [record.original_sentence for record in candidates] == [
        "Computer Science"
    ]


def test_initialize_corpus_extracts_into_the_given_directory(tmp_path):
    archive_path = tmp_path / "small.zip"
    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("a.txt", "line\n")

    destination = tmp_path / "extracted"
    initialize_corpus(str(archive_path), str(destination))

    assert (destination / "a.txt").exists()
    assert Path(archive_path).exists()
