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
from src.semantic.embedding_provider import embed_text
from src.semantic.index_builder import open_or_build_store
from src.semantic.search import semantic_search_store


LOGGER = get_application_logger()


def main() -> None:
    """Initialize the corpus once and start the interactive CLI."""
    configure_logging()
    LOGGER.info("Application started.")
    semantic_store = None

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

        try:
            semantic_store = open_or_build_store(index.records)
        except Exception:
            LOGGER.error("Semantic search initialization failed.")

        if semantic_store is None:
            run_cli()
        else:
            run_cli(
                semantic_search_fn=semantic_search_store,
                embedded_sentences=semantic_store,
                embedder=embed_text,
            )
    finally:
        try:
            if semantic_store is not None:
                semantic_store.close()
        finally:
            shutdown_logging()


if __name__ == "__main__":
    main()
