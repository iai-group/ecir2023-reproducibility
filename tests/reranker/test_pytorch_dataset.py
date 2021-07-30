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
    ranking1.add_doc("1", "doc1 content", 50.62)
    ranking1.add_doc("2", "doc2 content", 1.52)
    ranking1.add_doc("3", "doc3 content", 80.22)

    ranking2 = Ranking("1")
    ranking2.add_doc("1", "doc1 content", 20.43)
    ranking2.add_doc("4", "doc4 content", 12.3)
    ranking2.add_doc("5", "doc3 content", 100)

    return [ranking1, ranking2]


def test_populate_pairs(dummy_queries, dummy_rankings):
    dataset = PointWiseDataset(dummy_queries, dummy_rankings)
    assert len(dataset._query_doc_pairs) == 6
    assert (
        dataset._query_doc_pairs[0][0]
        == "How do you know when your garage door opener is going bad?"
    )
    assert dataset._query_doc_pairs[0][1] == "doc3 content"
