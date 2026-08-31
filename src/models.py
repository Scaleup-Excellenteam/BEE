from dataclasses import dataclass


@dataclass
class SentenceRecord:
    original_sentence: str
    normalized_sentence: str
    source_text: str
    offset: int
