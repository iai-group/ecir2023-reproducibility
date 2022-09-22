"""Tests for BERT and T5 rerankers."""
from treccast.core.base import Query
from treccast.core.ranking import Ranking
from treccast.core.util.fine_tuning.finetuning_data_loader import (
    FineTuningDataLoader,
)
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.simpletransformers_reranker_trainer import (
    SimpleTransformersTrainer,
)
from treccast.reranker.t5_reranker import T5Reranker


def test_bert_reranker(query: Query, ranking: Ranking) -> None:
    reranker = BERTReranker()
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0].doc_id == "3"
    assert reranking[1].doc_id == "1"


def test_t5_reranker(query: Query, ranking: Ranking) -> None:
    reranker = T5Reranker()
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0].doc_id == "3"
    assert reranking[1].doc_id == "1"


def test_bert_simpletransformer_reranker(
    query: Query, ranking: Ranking
) -> None:
    model_dir = "/tmp/simpletransformers_test/"
    train_args = SimpleTransformersTrainer.get_default_simpletransformers_args(
        model_dir
    )
    train_args["num_train_epochs"] = 1
    st_trainer = SimpleTransformersTrainer(
        base_model="bert",
        bert_type="nboost/pt-bert-base-uncased-msmarco",
        train_args=train_args,
    )
    fnt_loader = FineTuningDataLoader(
        "data/fine_tuning/trec_cast/Y1Y2_manual_qrels.tsv"
    )
    trec_queries, trec_rankings = fnt_loader.get_query_ranking_pairs()
    st_trainer.train(queries=trec_queries[:10], rankings=trec_rankings[:10])
    reranker = BERTReranker(
        base_model="bert-base-uncased",
        model_path="/tmp/simpletransformers_test/best_model",
    )
    reranking = reranker.rerank(query, ranking).fetch_topk_docs()
    assert len(reranking) == 3
    assert reranking[0].doc_id == "3"
    assert reranking[1].doc_id == "1"
