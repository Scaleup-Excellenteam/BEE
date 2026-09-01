"""Offline corpus stage: extraction, normalization, loading and indexing."""

from src.corpus.archive import extract_archive
from src.corpus.index import CorpusIndex
from src.corpus.initialization import initialize_corpus, load_or_initialize_corpus
from src.corpus.loader import load_corpus
from src.corpus.normalizer import normalize_text

__all__ = [
    "CorpusIndex",
    "extract_archive",
    "initialize_corpus",
    "load_or_initialize_corpus",
    "load_corpus",
    "normalize_text",
]
