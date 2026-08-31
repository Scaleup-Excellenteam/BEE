"""Command-line interface for Spec 3 autocomplete."""

from autocomplete import get_best_k_completions


def main() -> None:
    """Run the interactive autocomplete command-line interface."""
    current_input = ""

    print("The system is ready. Enter your text:")

    while True:
        user_input = input()

        if user_input == "#":
            current_input = ""
            continue

        current_input += user_input

        results = get_best_k_completions(current_input)

        for result in results:
            print(f"Completed sentence: {result.completed_sentence}")
            print(f"Source: {result.source_text}")
            print(f"Offset: {result.offset}")
            print(f"Score: {result.score}")
            print()


if __name__ == "__main__":
    main()
