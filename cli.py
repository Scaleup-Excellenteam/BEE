"""Command-line interface for Spec 3 autocomplete."""

import json
import sys
import time

from autocomplete import get_best_k_completions
from src.logging_config import get_application_logger
from src.translation import TranslationError, TranslationService


LOGGER = get_application_logger()

TRANSLATE_COMMAND = "/translate"
SPANISH_COMMAND = "/spanish"
ENGLISH_COMMAND = "/english"
TRANSLATION_MODE_ENABLED = (
    "Translation mode enabled. Target language: English."
)
SPANISH_MODE_ENABLED = "Spanish result localization enabled."
ENGLISH_MODE_ENABLED = "English mode enabled."
TRANSLATION_UNAVAILABLE = (
    "Translation mode is unavailable: no translation service configured."
)
SPANISH_LOCALIZATION_UNAVAILABLE = (
    "Spanish localization is unavailable; showing original results."
)
TRANSLATED_QUERY_PREFIX = "Translated English query:"
SPANISH_LANGUAGE_CODE = "es"


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


def main(translation_service: TranslationService | None = None) -> None:
    """Run the interactive autocomplete command-line interface."""
    current_input = ""
    translation_mode = False
    output_language = None

    print("The system is ready. Enter your text:")

    while True:
        try:
            user_input = _read_prefilled_input(current_input)
        except EOFError:
            LOGGER.info("Application closed by the user.")
            return
        except KeyboardInterrupt:
            LOGGER.info("Application interrupted by the user.")
            return

        if "#" in user_input:
            current_input = ""
            LOGGER.info(
                "The user finished the current query and started a new one."
            )
            continue

        command = user_input.strip().lower()
        if command == TRANSLATE_COMMAND:
            translation_mode = True
            output_language = None
            print(TRANSLATION_MODE_ENABLED)
            continue
        if command == SPANISH_COMMAND:
            translation_mode = False
            output_language = SPANISH_LANGUAGE_CODE
            print(SPANISH_MODE_ENABLED)
            continue
        if command == ENGLISH_COMMAND:
            translation_mode = False
            output_language = None
            print(ENGLISH_MODE_ENABLED)
            continue

        current_input = user_input
        logged_query = json.dumps(current_input, ensure_ascii=False)
        LOGGER.info("User submitted a search query: %s", logged_query)
        search_started = time.perf_counter()

        search_query = current_input
        if translation_mode:
            if translation_service is None:
                print(TRANSLATION_UNAVAILABLE)
                continue

            try:
                translation = translation_service.translate_to_english(
                    current_input
                )
            except TranslationError as error:
                LOGGER.warning("Translation failed: %s", error)
                print(f"Translation failed: {error}")
                continue

            search_query = translation.translated_text
            print(f"{TRANSLATED_QUERY_PREFIX} {search_query}")

        try:
            results = get_best_k_completions(search_query)
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

            if (
                output_language == SPANISH_LANGUAGE_CODE
                and translation_service is None
            ):
                print(SPANISH_LOCALIZATION_UNAVAILABLE)

            for position, result in enumerate(results, start=1):
                if output_language == SPANISH_LANGUAGE_CODE:
                    translated_sentence = None
                    if translation_service is not None:
                        try:
                            translation = translation_service.translate(
                                result.completed_sentence,
                                output_language,
                            )
                            translated_sentence = translation.translated_text
                        except TranslationError as error:
                            LOGGER.warning(
                                "Spanish localization failed for result "
                                "%d: %s",
                                position,
                                error,
                            )

                    print(
                        f"{position}. Original: "
                        f"{result.completed_sentence}"
                    )
                    if translated_sentence is None:
                        print("   Spanish: unavailable")
                    else:
                        print(f"   Spanish: {translated_sentence}")
                    print(f"   Source: {result.source_text}")
                    print(f"   Offset: {result.offset}")
                    print(f"   Score: {result.score}")
                    continue

                print(
                    f"{position}. {result.completed_sentence} "
                    f"({result.source_text}:{result.offset}, "
                    f"score={result.score})"
                )


if __name__ == "__main__":
    main()
