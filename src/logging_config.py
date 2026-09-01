"""Central file-logging configuration for the autocomplete application."""

from __future__ import annotations

import logging
from pathlib import Path


APPLICATION_LOGGER_NAME = "bee.autocomplete"
DEFAULT_LOG_PATH = Path("logs/autocomplete.log")


def get_application_logger() -> logging.Logger:
    """Return the shared application logger without configuring handlers."""
    return logging.getLogger(APPLICATION_LOGGER_NAME)


def configure_logging(
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> logging.Logger:
    """Configure and return the application's reusable file logger."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_application_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger


def shutdown_logging() -> None:
    """Flush and close application log handlers."""
    logger = get_application_logger()

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
