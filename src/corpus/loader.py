from pathlib import Path

from src.models import SentenceRecord
from src.corpus.normalizer import normalize_text


def load_corpus(root_path: str) -> list[SentenceRecord]:
    root = Path(root_path)
    records: list[SentenceRecord] = []

    for file_path in root.rglob("*.txt"):
        relative_path = file_path.relative_to(root)

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                original_sentence = line.rstrip("\r\n")

                if not original_sentence.strip():
                    continue

                normalized_sentence = normalize_text(original_sentence)

                record = SentenceRecord(
                    original_sentence=original_sentence,
                    normalized_sentence=normalized_sentence,
                    source_text=relative_path.as_posix(),
                    offset=line_number,
                )

                records.append(record)

    return records