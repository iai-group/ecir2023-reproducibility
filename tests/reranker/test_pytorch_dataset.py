import pytest

from treccast.core.ranking import Ranking
from treccast.reranker.train.pytorch_dataset import PointWiseDataset
from treccast.core.query.query import Query


@pytest.fixture
def dummy_queries():
    return [
        Query(
            "qid_0",
            "How do you know when your garage door opener is going bad?",
        ),
        Query(
            "qid_1",
            "How much does it cost for someone to repair a garage door opener?",
        ),
    ]


@pytest.fixture
def dummy_rankings():
    ranking1 = Ranking("0")
    ranking1.add_doc("1", 50.62, "doc1 content")
    ranking1.add_doc("2", 1.52, "doc2 content")
    ranking1.add_doc("3", 80.22, "doc3 content")

    ranking2 = Ranking("1")
    ranking2.add_doc("1", 20.43, "doc1 content")
    ranking2.add_doc("4", 12.3, "doc4 content")
    ranking2.add_doc("5", 100, "doc3 content")

    return [ranking1, ranking2]


def test_populate_pairs(dummy_queries, dummy_rankings):
    dataset = PointWiseDataset(dummy_queries, dummy_rankings)
    assert len(dataset._query_doc_pairs) == 6
    assert (
        dataset._query_doc_pairs[0][0]
        == "How do you know when your garage door opener is going bad?"
    )
    assert dataset._query_doc_pairs[0][1] == "doc3 content"
