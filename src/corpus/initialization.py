"""Offline initialization of the corpus.

This is the one-time startup path:

    Archive.zip
        -> safe extraction into extracted_archive/
        -> load_corpus()          (records in RAM)
        -> CorpusIndex(records)   (trigram index built once)
        -> return the index

The returned index is meant to be kept alive for the whole run and reused
for every query.  It must never be rebuilt per query.
"""

import shutil
from pathlib import Path

from src.corpus.archive import extract_archive
from src.corpus.index import CorpusIndex
from src.corpus.loader import load_corpus
from src.corpus.persistence import (
    CorpusSnapshotError,
    calculate_archive_sha256,
    load_corpus_index,
    save_corpus_index,
)
from src.logging_config import get_application_logger

# Deterministic extraction directory, already listed in .gitignore.
DEFAULT_EXTRACTION_PATH = "extracted_archive"
DEFAULT_SNAPSHOT_PATH = "corpus_index.sqlite3"

LOGGER = get_application_logger()


def initialize_corpus(
    archive_path: str,
    extraction_path: str = DEFAULT_EXTRACTION_PATH,
) -> CorpusIndex:
    """Extract, load and index the corpus, and return the ready index."""
    _clear_extraction_path(archive_path, extraction_path)
    extracted_root = extract_archive(archive_path, extraction_path)
    records = load_corpus(extracted_root)

    return CorpusIndex(records)


def load_or_initialize_corpus(
    archive_path: str,
    extraction_path: str = DEFAULT_EXTRACTION_PATH,
    snapshot_path: str = DEFAULT_SNAPSHOT_PATH,
) -> CorpusIndex:
    """Load a matching snapshot, or cold-build and best-effort persist one."""
    archive = Path(archive_path)
    if not archive.exists():
        # Preserve the cold path's established FileNotFoundError and message.
        return initialize_corpus(archive_path, extraction_path)

    archive_sha256 = calculate_archive_sha256(archive)
    try:
        return load_corpus_index(
            snapshot_path,
            expected_archive_sha256=archive_sha256,
        )
    except (CorpusSnapshotError, OSError) as error:
        LOGGER.info("Corpus snapshot unavailable; rebuilding: %s", error)

    index = initialize_corpus(archive_path, extraction_path)

    try:
        completed_archive_sha256 = calculate_archive_sha256(archive)
        if completed_archive_sha256 != archive_sha256:
            LOGGER.warning(
                "Archive changed during corpus initialization; snapshot skipped."
            )
            return index
        save_corpus_index(
            index,
            snapshot_path,
            archive_sha256=archive_sha256,
        )
    except Exception as error:
        # The snapshot is an optimization.  A valid cold-built index remains
        # usable even if cache creation fails.
        LOGGER.warning("Corpus snapshot could not be saved: %s", error)

    return index


def _clear_extraction_path(
    archive_path: str,
    extraction_path: str,
) -> None:
    """Remove the designated derived extraction tree before a cold build."""
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    extraction = Path(extraction_path)
    extraction_resolved = extraction.resolve()
    working_directory = Path.cwd().resolve()
    filesystem_root = Path(extraction_resolved.anchor)

    if (
        extraction_resolved == filesystem_root
        or extraction_resolved == working_directory
        or extraction_resolved in working_directory.parents
    ):
        raise ValueError(
            f"Unsafe extraction directory cannot be cleared: {extraction}"
        )

    archive_resolved = archive.resolve()
    if (
        archive_resolved == extraction_resolved
        or extraction_resolved in archive_resolved.parents
    ):
        raise ValueError(
            "Archive cannot be located inside the extraction directory"
        )

    if extraction.is_symlink():
        raise ValueError(
            f"Extraction directory cannot be a symbolic link: {extraction}"
        )
    if extraction.exists():
        if not extraction.is_dir():
            raise NotADirectoryError(
                f"Extraction path is not a directory: {extraction}"
            )
        shutil.rmtree(extraction)
