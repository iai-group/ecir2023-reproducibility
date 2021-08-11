"""Tests for SimpleTokenizer."""

from treccast.core.query.preprocessing.tokenizer import SimpleTokenizer


def test_empty_query():
    assert SimpleTokenizer.get_tokens("") == []


def test_single_term():
    assert SimpleTokenizer.get_tokens("abc") == ["abc"]


def test_question():
    assert SimpleTokenizer.get_tokens(
        "How does the DNA-based method work?"
    ) == [
        "dna-based",
        "method",
        "work",
    ]
