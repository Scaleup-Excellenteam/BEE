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
from src.logsearch.log_index import LogSearchService


LOGGER = get_application_logger()


def initialize_log_search() -> LogSearchService:
    """Return a ready semantic log search service.

    The service uses local semantic inference, so it needs no API key and
    no cloud service.  Once the model is installed locally, startup and
    search do not require a cloud API; only a machine that does not yet
    have the model downloads it, once.

    On a first run it imports the prepared historical fault dataset into
    the satellite history and builds the cache from every ERROR, WARNING
    and CRITICAL entry in it; afterwards it embeds only the faults
    appended since, which normally means none at all.
    """
    LOGGER.info("Semantic log search preparation started.")
    preparation_started = time.perf_counter()

    service = LogSearchService()
    added = service.refresh()

    # Load the model here rather than on the first query.  A cold cache
    # has already loaded it during refresh, so this costs nothing then;
    # a warm cache embeds nothing, and without this the first mode 2
    # query would pay several seconds of model loading on its own.
    service.warm_up()

    elapsed_seconds = time.perf_counter() - preparation_started
    indexed = len(service)

    LOGGER.info(
        "Semantic log search is ready with %d indexed fault records "
        "(%d newly embedded) in %.3f seconds.",
        indexed,
        added,
        elapsed_seconds,
    )

    # A warm cache shows a near zero time here and "Newly embedded: 0",
    # which is the whole argument for keeping the cache on disk.
    print("Semantic Log Search ready:")
    print(f"  Historical faults: {indexed}")
    print(f"  Newly embedded: {added}")
    print(f"  Initialization time: {elapsed_seconds:.3f} sec")
    print()

    return service


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
        # Log Search is optional: if the local model or its cache cannot
        # be prepared, the failure is logged and mode 1 carries on.
        try:
            log_search = initialize_log_search()
        except Exception as error:
            LOGGER.exception(
                "Semantic Log Search is unavailable because its "
                "preparation failed: %s",
                error,
            )
            log_search = None

        if log_search is None:
            run_cli()
        else:
            try:
                # The service is built once here and reused for every
                # mode 2 entry.  Mode 2 records: each message is matched
                # against the history and then becomes part of it, which
                # is why the CLI is given record_error rather than the
                # read-only search.
                run_cli(
                    record_fault_fn=log_search.record_error,
                    log_size_fn=lambda: len(log_search),
                )
            finally:
                log_search.close()
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
