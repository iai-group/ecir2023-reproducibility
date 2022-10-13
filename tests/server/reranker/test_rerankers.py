"""Tests for T5 rerankers."""
from treccast.core.base import Query
from treccast.core.ranking import Ranking
from treccast.reranker.t5_reranker import T5Reranker


def test_t5_reranker(query: Query, ranking: Ranking) -> None:
    reranker = T5Reranker()
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0].doc_id == "3"
    assert reranking[1].doc_id == "1"
