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

from src.corpus.archive import extract_archive
from src.corpus.index import CorpusIndex
from src.corpus.loader import load_corpus

# Deterministic extraction directory, already listed in .gitignore.
DEFAULT_EXTRACTION_PATH = "extracted_archive"


def initialize_corpus(
    archive_path: str,
    extraction_path: str = DEFAULT_EXTRACTION_PATH,
) -> CorpusIndex:
    """Extract, load and index the corpus, and return the ready index."""
    extracted_root = extract_archive(archive_path, extraction_path)
    records = load_corpus(extracted_root)

    return CorpusIndex(records)
