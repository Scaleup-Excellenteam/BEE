from dataclasses import dataclass


@dataclass
class EmbeddedSentence:
    sentence: str
    source_text: str
    offset: int
    embedding: list[float]


@dataclass
class SemanticResult:
    sentence: str
    source_text: str
    offset: int
    similarity: float
