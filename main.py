"""Application startup for the integrated autocomplete system."""

import argparse
import importlib
import json
import time

from autocomplete import set_corpus_index
from cli import run_mode_menu as run_cli
from src.corpus.initialization import load_or_initialize_corpus

from src.logging_config import (
    configure_logging,
    get_application_logger,
    shutdown_logging,
)
from src.logsearch.log_index import LogSearchService
from src.translation import GoogleTranslationService, TranslationService


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
    # entry would pay several seconds of model loading on its own.
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


def _initialize_translation_service() -> TranslationService | None:
    """Return an available Translation service without blocking Part A."""
    try:
        service = GoogleTranslationService()
        if not service.project_id:
            LOGGER.warning(
                "Translation is unavailable because "
                "GOOGLE_CLOUD_PROJECT is not configured."
            )
            return None

        importlib.import_module("google.cloud.translate_v3")
    except ImportError:
        LOGGER.warning(
            "Translation is unavailable because google-cloud-translate "
            "is not installed."
        )
        return None
    except Exception as error:
        LOGGER.warning(
            "Translation service initialization failed (%s). "
            "Part A remains available.",
            type(error).__name__,
        )
        return None

    LOGGER.info("Translation service is configured and ready.")
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
            index = load_or_initialize_corpus(args.archive_path)
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

        translation_service = _initialize_translation_service()

        # Two independent optional features.  Each contributes its own
        # keyword arguments, so the CLI can lose one of them without
        # losing the other, and Part A runs even without both.
        cli_arguments = {}

        if log_search is not None:
            # The service is built once here and reused for every mode 2
            # entry.  Mode 2 records: each message is matched against the
            # history and then becomes part of it, which is why the CLI
            # is given record_error rather than the read-only search.
            cli_arguments.update(
                record_fault_fn=log_search.record_error,
                log_size_fn=lambda: len(log_search),
                storage_status_fn=log_search.storage_status,
            )

        if translation_service is not None:
            cli_arguments["translation_service"] = translation_service

        try:
            run_cli(**cli_arguments)
        finally:
            if log_search is not None:
                log_search.close()
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
