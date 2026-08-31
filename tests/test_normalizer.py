from src.corpus.normalizer import normalize_text


def test_lowercases_text():
    assert normalize_text("Hello World") == "hello world"


def test_punctuation_becomes_whitespace():
    assert normalize_text("hello,world") == "hello world"


def test_collapses_repeated_whitespace():
    assert normalize_text("hello   \t world") == "hello world"


def test_strips_surrounding_whitespace():
    assert normalize_text("   hello world   ") == "hello world"


def test_empty_and_blank_input():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_digits_are_preserved():
    assert normalize_text("Python 3.5.0 alpha 1") == "python 3 5 0 alpha 1"


def test_is_idempotent():
    once = normalize_text("  Computer, Science!  ")
    assert normalize_text(once) == once
