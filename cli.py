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

        if not results:
            print("No suggestions found.")
            continue

        print(f"Here are {len(results)} suggestions:")

        for position, result in enumerate(results, start=1):
            print(
                f"{position}. {result.completed_sentence} "
                f"({result.source_text}:{result.offset}, "
                f"score={result.score})"
            )


if __name__ == "__main__":
    main()
