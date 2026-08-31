"""Application startup for the integrated autocomplete system."""

import argparse

from autocomplete import set_corpus_index
from cli import main as run_cli
from src.corpus.initialization import initialize_corpus


def main() -> None:
    """Initialize the corpus once and start the interactive CLI."""
    parser = argparse.ArgumentParser(description="Run the autocomplete CLI")
    parser.add_argument("archive_path", help="Path to the corpus ZIP archive")
    args = parser.parse_args()

    index = initialize_corpus(args.archive_path)
    set_corpus_index(index)
    run_cli()


if __name__ == "__main__":
    main()
