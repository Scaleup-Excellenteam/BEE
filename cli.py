"""Command-line interface for Spec 3 autocomplete."""

import sys

from autocomplete import get_best_k_completions


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


def main() -> None:
    """Run the interactive autocomplete command-line interface."""
    current_input = ""

    print("The system is ready. Enter your text:")

    while True:
        try:
            user_input = _read_prefilled_input(current_input)
        except (EOFError, KeyboardInterrupt):
            return

        if "#" in user_input:
            current_input = ""
            continue

        current_input = user_input

        results = get_best_k_completions(current_input)

        if not results:
            print("No suggestions found.")
        else:
            print(f"Here are {len(results)} suggestions:")

            for position, result in enumerate(results, start=1):
                print(
                    f"{position}. {result.completed_sentence} "
                    f"({result.source_text}:{result.offset}, "
                    f"score={result.score})"
                )


if __name__ == "__main__":
    main()
