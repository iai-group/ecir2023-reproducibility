"""Tests Ranking class from Ranker"""
import csv
from io import StringIO

from treccast.core.ranking import Ranking
from treccast.core.scored_document import ScoredDocument


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


def test_add_docs():
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


def test_write_to_tsv_file():
    outfile = StringIO()
    tsv_writer = csv.writer(outfile, delimiter="\t")
    tsv_writer.writerow(["query_id", "query", "passage_id", "passage", "label"])
    ranking = Ranking(
        "123",
        [
            {"doc_id": "001", "score": 10, "content": "doc001\t content"},
            {"doc_id": "003", "score": 1, "content": "doc003 content"},
            {"doc_id": "002", "score": 5, "content": "doc002 content"},
        ],
    )
    ranking.write_to_tsv_file(tsv_writer, "test", k=2)

    outfile.seek(0)
    read_csv = list(csv.reader(outfile, delimiter="\t"))
    expected = [
        ["query_id", "query", "passage_id", "passage", "label"],
        ["123", "test", "001", "doc001\t content", "10"],
        ["123", "test", "002", "doc002 content", "5"],
    ]
    assert expected == read_csv


def test_write_to_trec_file():
    outfile = StringIO()
    ranking = Ranking(
        "123",
        [
            {"doc_id": "001", "score": 10},
            {"doc_id": "003", "score": 1},
            {"doc_id": "002", "score": 5},
        ],
    )
    ranking.write_to_trec_file(outfile, "test", k=2)
    outfile.seek(0)
    content = outfile.read()
    assert content == "123 Q0 001 1 10 test\n123 Q0 002 2 5 test\n"


def test_document_dataclass():
    doc = ScoredDocument(doc_id="1", content="Test document", score=1.0)
    assert doc.content == "Test document"
    assert doc.doc_id == "1"
    assert doc.score == 1.0

    doc_without_content = doc = ScoredDocument(doc_id="2", score=1.02)
    assert doc_without_content.content is None
    assert doc_without_content.doc_id == "2"
    assert doc_without_content.score == 1.02


def test_load_rankings_from_tsv_file_num_queries():
    path = "tests/data/ranking_sample_1.tsv"
    rankings = Ranking.load_rankings_from_tsv_file(path)
    assert len(rankings) == 2


def test_load_rankings_from_tsv_file_first_query():
    path = "tests/data/ranking_sample_1.tsv"
    rankings = Ranking.load_rankings_from_tsv_file(path)
    first_ranking = rankings["001"]
    assert first_ranking.query_id == "001"
    assert len(first_ranking) == 3


def test_load_rankings_from_tsv_file_passage_text():
    path = "tests/data/ranking_sample_1.tsv"
    rankings = Ranking.load_rankings_from_tsv_file(path)
    ids, contents = rankings["002"].documents()
    assert ids == ["002", "004", "005"]
    assert contents == ["test passage 2", "test passage 4", "test passage 5"]
