"""Command-line interface for Spec 3 autocomplete."""

import json
import sys
import time

from autocomplete import get_best_k_completions
from src.logging_config import get_application_logger


LOGGER = get_application_logger()

# Typed inside a mode to return to the main menu.  It is navigation only:
# it is never passed to autocomplete or to semantic search.
BACK_COMMAND = "back"


def _is_back_command(user_input: str) -> bool:
    """Return whether the user asked to return to the main menu."""
    return user_input.strip().lower() == BACK_COMMAND


def _print_mode_menu() -> None:
    """Display the application's top-level mode choices."""
    print("=" * 32)
    print("       SMART AUTOCOMPLETE")
    print("=" * 32)
    print()
    print("Choose mode:")
    print()
    print("1. Regular Autocomplete")
    print("2. Semantic Search")
    print("3. Exit")
    print()


def _run_regular_mode() -> bool:
    """Run Part A autocomplete until the user leaves the mode.

    Returns ``True`` when the user asked to go back to the main menu, and
    ``False`` when the input stream ended and the application should stop.
    """
    print("Regular Autocomplete")
    print(f"Type '{BACK_COMMAND}' to return to the main menu.")
    print()

    return main()


def _run_semantic_mode(
    semantic_search_fn,
    embedded_sentences,
    embedder,
) -> bool:
    """Answer semantic queries until the user leaves the mode.

    Returns ``True`` when the user asked to go back to the main menu, and
    ``False`` when the input stream ended and the application should stop.
    """
    print("Semantic Search")

    if (
        semantic_search_fn is None
        or embedded_sentences is None
        or embedder is None
    ):
        print()
        print("Semantic Search is temporarily unavailable.")
        return True

    print(f"Type '{BACK_COMMAND}' to return to the main menu.")
    print()

    while True:
        print("Enter your query:")

        try:
            query = input()
        except EOFError:
            LOGGER.info("Application closed by the user.")
            return False
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted by the user.")
            return False

        if _is_back_command(query):
            LOGGER.info("The user returned to the main menu.")
            return True

        _print_semantic_results(
            query,
            semantic_search_fn,
            embedded_sentences,
            embedder,
        )


def _print_semantic_results(
    query,
    semantic_search_fn,
    embedded_sentences,
    embedder,
) -> None:
    """Run one semantic query and print whatever it returns."""
    try:
        results = semantic_search_fn(
            query,
            embedded_sentences,
            embedder,
        )
    except Exception:
        LOGGER.error("Semantic search failed.")
        print("Semantic Search is temporarily unavailable.")
        return

    if not results:
        print("No semantic results found.")
        return

    print(f"Here are {len(results)} semantic results:")
    print()

    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.sentence}")
        print(f"   Source: {result.source_text}:{result.offset}")
        print(f"   Similarity: {result.similarity:.3f}")
        print()


def run_mode_menu(
    *,
    semantic_search_fn=None,
    embedded_sentences=None,
    embedder=None,
) -> None:
    """Run the mode menu with optional injected semantic dependencies."""
    while True:
        try:
            _print_mode_menu()
            choice = input("> ").strip()

            if choice == "1":
                if not _run_regular_mode():
                    return

                continue

            if choice == "2":
                if not _run_semantic_mode(
                    semantic_search_fn,
                    embedded_sentences,
                    embedder,
                ):
                    return

                continue

            if choice == "3":
                return

            print("Invalid option. Please choose 1, 2, or 3.")
        except EOFError:
            LOGGER.info("Application closed by the user.")
            return
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted by the user.")
            return


def _read_windows_prefilled_input(initial_text: str) -> str:
    """Read an editable line prefilled with text in a Windows console."""
    import msvcrt

    characters = list(initial_text)
    cursor = len(characters)
    sys.stdout.write(initial_text)
    sys.stdout.flush()

    while True:
        try:
            character = msvcrt.getwch()
        except KeyboardInterrupt:
            print()
            raise

        if character in ("\r", "\n"):
            print()
            return "".join(characters)

        if character == "\x03":
            print()
            raise KeyboardInterrupt

        if character in ("\x04", "\x1a"):
            print()
            raise EOFError

        if character == "\b":
            if cursor == 0:
                continue

            del characters[cursor - 1]
            cursor -= 1
            suffix = "".join(characters[cursor:])
            sys.stdout.write("\b" + suffix + " ")
            sys.stdout.write("\b" * (len(suffix) + 1))
            sys.stdout.flush()
            continue

        if character in ("\x00", "\xe0"):
            key = msvcrt.getwch()

            if key == "K" and cursor > 0:  # Left arrow.
                sys.stdout.write("\b")
                cursor -= 1
            elif key == "M" and cursor < len(characters):  # Right arrow.
                sys.stdout.write(characters[cursor])
                cursor += 1
            elif key == "G" and cursor > 0:  # Home.
                sys.stdout.write("\b" * cursor)
                cursor = 0
            elif key == "O" and cursor < len(characters):  # End.
                sys.stdout.write("".join(characters[cursor:]))
                cursor = len(characters)
            elif key == "S" and cursor < len(characters):  # Delete.
                del characters[cursor]
                suffix = "".join(characters[cursor:])
                sys.stdout.write(suffix + " ")
                sys.stdout.write("\b" * (len(suffix) + 1))

            sys.stdout.flush()
            continue

        if not character.isprintable():
            continue

        characters.insert(cursor, character)
        suffix = "".join(characters[cursor:])
        sys.stdout.write(suffix)
        cursor += 1
        sys.stdout.write("\b" * (len(suffix) - 1))
        sys.stdout.flush()


def _read_prefilled_input(initial_text: str) -> str:
    """Read a full editable line starting with the accumulated text."""
    if not initial_text or not sys.stdin.isatty():
        return input()

    if sys.platform == "win32":
        # input() cannot seed Windows' editable buffer; msvcrt is the
        # standard-library console interface available on Python 3.12.
        return _read_windows_prefilled_input(initial_text)

    try:
        import readline
    except ImportError:
        return input()

    readline.set_startup_hook(lambda: readline.insert_text(initial_text))
    try:
        return input()
    finally:
        readline.set_startup_hook()


def main() -> bool:
    """Run the interactive autocomplete command-line interface.

    Returns ``True`` when the user asked to go back to the main menu, and
    ``False`` when the input stream ended and the application should stop.
    """
    current_input = ""

    print("The system is ready. Enter your text:")

    while True:
        try:
            user_input = _read_prefilled_input(current_input)
        except EOFError:
            LOGGER.info("Application closed by the user.")
            return False
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted by the user.")
            return False

        # Navigation is checked first so that "back" never reaches the
        # autocomplete engine.  It cannot contain "#", so the reset rule
        # below is unaffected.
        if _is_back_command(user_input):
            LOGGER.info("The user returned to the main menu.")
            return True

        if "#" in user_input:
            current_input = ""
            LOGGER.info(
                "The user finished the current query and started a new one."
            )
            continue

        current_input = user_input
        logged_query = json.dumps(current_input, ensure_ascii=False)
        LOGGER.info("User submitted a search query: %s", logged_query)
        search_started = time.perf_counter()

        try:
            results = get_best_k_completions(current_input)
        except Exception as error:
            elapsed_seconds = time.perf_counter() - search_started
            LOGGER.exception(
                "An error occurred while searching for %s after "
                "%.3f seconds: %s",
                logged_query,
                elapsed_seconds,
                error,
            )
            raise

        elapsed_seconds = time.perf_counter() - search_started

        if not results:
            LOGGER.info(
                "Search completed successfully in %.3f seconds. "
                "No suggestions were found for the current query.",
                elapsed_seconds,
            )
            print("No suggestions found.")
        else:
            suggestion_count = len(results)
            suggestion_word = (
                "suggestion" if suggestion_count == 1 else "suggestions"
            )
            return_verb = "was" if suggestion_count == 1 else "were"
            LOGGER.info(
                "Search completed successfully in %.3f seconds. "
                "%d %s %s returned.",
                elapsed_seconds,
                suggestion_count,
                suggestion_word,
                return_verb,
            )
            print(f"Here are {len(results)} suggestions:")

            for position, result in enumerate(results, start=1):
                print(
                    f"{position}. {result.completed_sentence} "
                    f"({result.source_text}:{result.offset}, "
                    f"score={result.score})"
                )


if __name__ == "__main__":
    run_mode_menu()
