import re
import string


def normalize_text(text: str) -> str:
    text = text.lower()

    punctuation_table = str.maketrans(
        string.punctuation,
        " " * len(string.punctuation),
    )
    text = text.translate(punctuation_table)

    text = re.sub(r"\s+", " ", text)

    return text.strip()