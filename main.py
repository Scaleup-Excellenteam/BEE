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


LOGGER = get_application_logger()


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
        run_cli()
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
