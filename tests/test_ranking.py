"""Tests Ranking class from Ranker"""
import pytest

from treccast.core.ranking import Ranking


@pytest.fixture
def empty_ranker():
    return Ranking("0")


def test_empty_ranker(empty_ranker):
    assert empty_ranker.fetch_topk_docs(1) == []


def test_add_docs(empty_ranker):
    empty_ranker.add_docs([("3", 80.22), ("1", 50.62), ("2", 1.52)])
    assert empty_ranker.get_doc_score("1") == 50.62
    assert empty_ranker.get_doc_score("2") == 1.52
    assert empty_ranker.get_doc_score("3") == 80.22


def test_add_doc(empty_ranker):
    empty_ranker.set_doc_score("1", 50.62)
    assert empty_ranker.get_doc_score("1") == 50.62


def test_update_doc(empty_ranker):
    empty_ranker.set_doc_score("1", 50.62)
    empty_ranker.set_doc_score("1", 1.52)
    assert empty_ranker.get_doc_score("1") == 1.52


def test_fetch_topk_docs(empty_ranker):
    empty_ranker.set_doc_score("1", 50.62)
    empty_ranker.set_doc_score("2", 1.52)
    empty_ranker.set_doc_score("3", 80.22)
    assert empty_ranker.fetch_topk_docs(2) == [("3", 80.22), ("1", 50.62)]
