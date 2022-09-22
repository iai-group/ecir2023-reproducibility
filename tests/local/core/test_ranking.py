"""Tests Ranking class from Ranker"""
import csv
from io import StringIO

from treccast.core.base import ScoredDocument
from treccast.core.ranking import Ranking


def test_empty_ranking():
    ranking = Ranking("0")
    assert ranking.fetch_topk_docs(1) == []
    assert len(ranking) == 0


def test_add_doc():
    ranking = Ranking("1")
    ranking.add_doc(ScoredDocument("3", "doc3 content", 80.22))
    ranking.add_doc(ScoredDocument("1", "doc1 content", 50.62))
    ranking.add_doc(ScoredDocument("2", "doc2 content", 1.52))
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
            ScoredDocument("001", score=10),
            ScoredDocument("003", score=1),
            ScoredDocument("002", score=5),
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


def test_update():
    ranking = Ranking("1")
    ranking.add_docs(
        [
            ScoredDocument("001", "test content doc 1", 10),
            ScoredDocument("003", score=1),
        ]
    )
    assert len(ranking) == 2
    ranking.update(
        [
            ScoredDocument("001", "test duplicate content doc 1", 3),
            ScoredDocument("004", score=5),
        ]
    )
    assert len(ranking) == 3
    doc_ids, contents = ranking.documents()
    assert doc_ids == ["001", "003", "004"]
    assert contents == [
        "test content doc 1",
        None,
        None,
    ]


def test_fetch_topk_docs():
    ranking = Ranking("2")
    ranking.add_doc(ScoredDocument("1", "doc1 content", 50.62))
    ranking.add_doc(ScoredDocument("2", "doc2 content", 1.52))
    ranking.add_doc(ScoredDocument("3", "doc3 content", 80.22))
    top2docs = ranking.fetch_topk_docs(2)
    assert top2docs[0].doc_id == "3"
    assert top2docs[1].score == 50.62


def test_fetch_topk_docs_unique():
    ranking = Ranking("2")
    ranking.add_doc(ScoredDocument("1", "doc1 content", 50.62))
    ranking.add_doc(ScoredDocument("1", "doc1 content", 1.52))
    ranking.add_doc(ScoredDocument("3", "doc3 content", 80.22))
    topkdocs = ranking.fetch_topk_docs(unique=True)
    assert len(topkdocs) == 2
    assert topkdocs[0].content == "doc3 content"
    assert topkdocs[1].doc_id == "1"


def test_write_to_tsv_file():
    outfile = StringIO()
    tsv_writer = csv.writer(outfile, delimiter="\t")
    tsv_writer.writerow(["query_id", "query", "passage_id", "passage", "label"])
    ranking = Ranking(
        "123",
        [
            ScoredDocument("001", "doc001\t content", 10),
            ScoredDocument("003", "doc003 content", 1),
            ScoredDocument("002", "doc002 content", 5),
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
            ScoredDocument("001", score=10),
            ScoredDocument("003", score=1),
            ScoredDocument("002", score=5),
            ScoredDocument("002", score=6),
        ],
    )
    ranking.write_to_trec_file(outfile, "test", k=3)
    outfile.seek(0)
    content = outfile.read()
    assert (
        content
        == "123 Q0 001 1 10 test\n123 Q0 002 2 6 test\n123 Q0 003 3 1 test\n"
    )


def test_load_rankings_from_tsv_file_num_queries():
    path = "tests/data/ranking_sample_1.tsv"
    _, rankings = Ranking.load_rankings_from_tsv_file(path)
    assert len(rankings) == 2


def test_load_rankings_from_tsv_file_first_query():
    path = "tests/data/ranking_sample_1.tsv"
    queries, _ = Ranking.load_rankings_from_tsv_file(path)
    first_query = queries["001"]
    assert first_query.query_id == "001"
    assert first_query.question == "test query 1"


def test_load_rankings_from_tsv_file_second_ranking():
    path = "tests/data/ranking_sample_1.tsv"
    _, rankings = Ranking.load_rankings_from_tsv_file(path)
    second_ranking = rankings["002"]
    assert second_ranking.query_id == "002"
    assert len(second_ranking) == 3


def test_load_rankings_from_tsv_file_passage_text():
    path = "tests/data/ranking_sample_1.tsv"
    _, rankings = Ranking.load_rankings_from_tsv_file(path)
    ids, contents = rankings["002"].documents()
    assert ids == ["002", "004", "005"]
    assert contents == ["test passage 2", "test passage 4", "test passage 5"]


def test_load_rankings_from_runfile():
    path = "tests/data/ranking_sample_1.trec"
    rankings = Ranking.load_rankings_from_runfile(path)
    ids, contents = rankings["123"].documents()
    assert ids == ["001", "002", "003"]
    assert contents == [None] * 3
