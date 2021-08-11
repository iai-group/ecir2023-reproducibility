"""Tests Ranking class from Ranker"""
from io import StringIO

from treccast.core.ranking import Ranking


def test_empty_ranking():
    ranking = Ranking("0")
    assert ranking.fetch_topk_docs(1) == []
    assert len(ranking) == 0


def test_add_doc():
    ranking = Ranking("1")
    ranking.add_doc("3", 80.22, "doc3 content")
    ranking.add_doc("1", 50.62, "doc1 content")
    ranking.add_doc("2", 1.52, "doc2 content")
    assert len(ranking) == 3
    doc_ids, contents = ranking.documents()
    assert doc_ids == ["3", "1", "2"]
    assert contents == [
        "doc3 content",
        "doc1 content",
        "doc2 content",
    ]


def test_add_multiple_docs():
    ranking = Ranking("1")
    ranking.add_docs(
        [
            {"doc_id": "001", "score": 10},
            {"doc_id": "003", "score": 1},
            {"doc_id": "002", "score": 5},
        ]
    )
    assert len(ranking) == 3
    doc_ids, contents = ranking.documents()
    assert doc_ids == ["001", "003", "002"]
    assert contents == [
        None,
        None,
        None,
    ]


def test_fetch_topk_docs():
    ranking = Ranking("2")
    ranking.add_doc("1", 50.62, "doc1 content")
    ranking.add_doc("2", 1.52, "doc2 content")
    ranking.add_doc("3", 80.22, "doc3 content")
    top2docs = ranking.fetch_topk_docs(2)
    assert top2docs[0] == {
        "doc_id": "3",
        "score": 80.22,
        "content": "doc3 content",
    }
    assert top2docs[1] == {
        "doc_id": "1",
        "score": 50.62,
        "content": "doc1 content",
    }


def test_write_to_file():
    outfile = StringIO()
    ranking = Ranking(
        "123",
        [
            {"doc_id": "001", "score": 10},
            {"doc_id": "003", "score": 1},
            {"doc_id": "002", "score": 5},
        ],
    )
    ranking.write_to_file(outfile, "test", k=2)
    outfile.seek(0)
    content = outfile.read()
    assert content == "123 Q0 001 1 10 test\n123 Q0 002 2 5 test\n"
