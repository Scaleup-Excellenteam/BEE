"""Application startup for the integrated autocomplete system."""

import argparse
import json
import time

from autocomplete import set_corpus_index
from cli import run_mode_menu as run_cli
from src.corpus.initialization import initialize_corpus
from src.logging_config import (
    configure_logging,
    get_application_logger,
    shutdown_logging,
)
from src.semantic.embedding_provider import embed_text, embed_texts
from src.semantic.index_builder import open_or_build_store
from src.semantic.search import semantic_search


LOGGER = get_application_logger()

# Semantic Search is still a smoke test, so it is built from a small
# sample of the corpus only.  Embedding all 2.58 million sentences would
# cost roughly 26,000 Gemini requests and about 8 GB of cache, which is
# not something application startup should do.
SEMANTIC_SAMPLE_SIZE = 100

# Deliberately NOT the production "embedding_cache" directory, so this
# sample can never be mistaken for a real full corpus build.
SEMANTIC_CACHE_PATH = "embedding_cache_test"


def initialize_semantic_store(index):
    """Return an embedding store built from a sample of the corpus.

    The sample is spread evenly across the whole corpus rather than taken
    from the front, so it covers every source file instead of only the
    first one.
    """
    step = max(1, len(index.records) // SEMANTIC_SAMPLE_SIZE)
    records = index.records[::step][:SEMANTIC_SAMPLE_SIZE]

    LOGGER.info(
        "Semantic preparation started using %d corpus sentences.",
        len(records),
    )

    store = open_or_build_store(
        records,
        cache_path=SEMANTIC_CACHE_PATH,
        batch_embedder=embed_texts,
    )

    LOGGER.info(
        "Semantic preparation completed successfully with %d sentences.",
        len(store),
    )

    return store


def main() -> None:
    """Initialize the corpus once and start the interactive CLI."""
    configure_logging()
    LOGGER.info("Application started.")

    try:
        parser = argparse.ArgumentParser(
            description="Run the autocomplete CLI"
        )
        parser.add_argument(
            "archive_path",
            help="Path to the corpus ZIP archive",
        )
        args = parser.parse_args()

        archive_name = json.dumps(args.archive_path, ensure_ascii=False)
        LOGGER.info(
            "Corpus preparation started using %s.",
            archive_name,
        )
        initialization_started = time.perf_counter()

        try:
            index = initialize_corpus(args.archive_path)
        except Exception as error:
            elapsed_seconds = time.perf_counter() - initialization_started
            LOGGER.exception(
                "An error occurred while preparing the corpus after "
                "%.3f seconds: %s",
                elapsed_seconds,
                error,
            )
            raise

        elapsed_seconds = time.perf_counter() - initialization_started
        LOGGER.info(
            "Corpus preparation completed successfully in %.3f seconds.",
            elapsed_seconds,
        )

        set_corpus_index(index)
        LOGGER.info("The autocomplete system is ready for searches.")

        # Regular Autocomplete is already usable at this point.  Semantic
        # Search is optional extra credit: if Gemini or the cache is not
        # available, the failure is logged and mode 1 carries on working.
        try:
            semantic_store = initialize_semantic_store(index)
        except Exception as error:
            LOGGER.exception(
                "Semantic Search is unavailable because its preparation "
                "failed: %s",
                error,
            )
            semantic_store = None

        if semantic_store is None:
            run_cli()
        else:
            try:
                run_cli(
                    semantic_search_fn=semantic_search,
                    embedded_sentences=semantic_store,
                    embedder=embed_text,
                )
            finally:
                semantic_store.close()
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
