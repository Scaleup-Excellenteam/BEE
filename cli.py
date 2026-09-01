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
    print("2. Semantic Log Search")
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


def _run_log_search_mode(record_fault_fn, log_size_fn=None) -> bool:
    """Record incoming faults until the user leaves the mode.

    Every message entered here is treated as a REAL satellite fault: it
    is compared against the history first, then becomes part of it.

    Returns ``True`` when the user asked to go back to the main menu, and
    ``False`` when the input stream ended and the application should stop.
    """
    print("Semantic Log Search")

    if record_fault_fn is None:
        print()
        print("Semantic Log Search is temporarily unavailable.")
        return True

    print(f"Type '{BACK_COMMAND}' to return to the main menu.")
    print("Each message entered is recorded as a new satellite fault.")
    print()

    while True:
        print("Enter an error/message:")

        try:
            query = input()
        except EOFError:
            LOGGER.info("Application closed by the user.")
            return False
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted by the user.")
            return False

        # Navigation is not a fault, and neither is an empty line.
        if _is_back_command(query):
            LOGGER.info("The user returned to the main menu.")
            return True

        if not query.strip():
            print("Please enter a fault message.")
            print()
            continue

        _record_and_report_fault(query, record_fault_fn, log_size_fn)


def _format_log_message(message: str) -> list[str]:
    """Return the display lines for one stored log message.

    A logged exception carries its traceback, which would fill the screen
    five times over.  Only the first line is shown, with a note saying
    how much was left out.
    """
    lines = message.splitlines() or [""]

    if len(lines) == 1:
        return lines

    return [lines[0], f"   ({len(lines) - 1} more line(s) in this entry)"]


def _print_query_metrics(elapsed_ms, searched, returned) -> None:
    """Print the cost of one query, in the units a reader cares about."""
    print("Query completed:")
    print(f"  Search time: {elapsed_ms:.1f} ms")

    if searched is not None:
        print(f"  Historical faults searched: {searched}")

    print(f"  Results returned: {returned}")
    print()


def _record_and_report_fault(message, record_fault_fn, log_size_fn=None) -> None:
    """Record one incoming fault and print the faults it resembles.

    The history size is read BEFORE recording, so the reported figure is
    the history the fault was actually compared against rather than the
    history it then became part of.

    The timer spans the whole call, so it covers embedding the message,
    scanning the cache and storing the new fault: that total is what a
    user actually waits for.
    """
    searched = log_size_fn() if log_size_fn is not None else None
    started = time.perf_counter()

    try:
        results = record_fault_fn(message)
    except Exception:
        LOGGER.error("Semantic log search failed.")
        print("Semantic Log Search is temporarily unavailable.")
        return

    elapsed_ms = (time.perf_counter() - started) * 1000

    LOGGER.info(
        "Satellite fault %s matched in %.1f ms against %s historical "
        "fault records, returning %d results.",
        json.dumps(message, ensure_ascii=False),
        elapsed_ms,
        "an unknown number of" if searched is None else searched,
        len(results),
    )

    if not results:
        print("No similar historical faults found.")
        print()
    else:
        print(f"Top {len(results)} similar historical faults:")
        print()

        for position, result in enumerate(results, start=1):
            message, *extra = _format_log_message(result.sentence)
            print(f"{position}. {message}")

            for line in extra:
                print(line)

            print(f"   Source: {result.source_text}:{result.offset}")
            print(f"   Similarity: {result.similarity:.2f}")
            print()

    _print_query_metrics(elapsed_ms, searched, len(results))


def run_mode_menu(*, record_fault_fn=None, log_size_fn=None) -> None:
    """Run the mode menu with an optional injected fault recorder.

    ``record_fault_fn`` is called as ``record_fault_fn(message)``.  It
    searches the history, returns the matching ``SemanticResult``
    objects, and only THEN stores the message as a new fault, so a
    message can never be returned as a match for itself.

    ``log_size_fn`` is called with no arguments and returns how many
    historical faults are indexed.  It is used only to report how much
    was searched, and the mode works without it.
    """
    while True:
        try:
            _print_mode_menu()
            choice = input("> ").strip()

            if choice == "1":
                if not _run_regular_mode():
                    return

                continue

            if choice == "2":
                if not _run_log_search_mode(record_fault_fn, log_size_fn):
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
