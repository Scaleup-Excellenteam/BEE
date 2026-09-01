"""Application startup for the integrated autocomplete system."""

import argparse
import importlib
import json
import time

from autocomplete import set_corpus_index
from cli import main as run_cli
from src.corpus.initialization import initialize_corpus
from src.logging_config import (
    configure_logging,
    get_application_logger,
    shutdown_logging,
)
from src.translation import GoogleTranslationService, TranslationService


LOGGER = get_application_logger()


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
        translation_service = _initialize_translation_service()
        if translation_service is None:
            run_cli()
        else:
            run_cli(translation_service=translation_service)
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
