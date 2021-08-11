"""Tests for BERT and T5 rerankers."""
from treccast.core.ranking import Ranking
from treccast.core.query.query import Query
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.t5_reranker import T5Reranker


def test_bert_reranker(query: Query, ranking: Ranking) -> None:
    reranker = BERTReranker()
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0]["doc_id"] == "3"
    assert reranking[1]["doc_id"] == "1"


def test_t5_reranker(query: Query, ranking: Ranking) -> None:
    reranker = T5Reranker()
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0]["doc_id"] == "3"
    assert reranking[1]["doc_id"] == "1"
