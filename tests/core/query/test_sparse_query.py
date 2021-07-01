"""Tests for SparseQuery."""

from treccast.core.query.sparse_query import SparseQuery


def test_empty_query():
    sq = SparseQuery("1", "")
    assert sq.terms == []


def test_single_term():
    sq = SparseQuery("2", "abc")
    assert sq.terms == ["abc"]


def test_question():
    sq = SparseQuery("3", "How does the DNA-based method work?")
    assert sq.terms == ["dna-based", "method", "work"]
