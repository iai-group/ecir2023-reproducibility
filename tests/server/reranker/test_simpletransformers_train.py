import pytest

from treccast.reranker.simpletransformers_reranker_trainer import (
    SimpleTransformersTrainer,
)
from treccast.core.util.fine_tuning.finetuning_data_loader import (
    FineTuningDataLoader,
)


@pytest.mark.order1
def test_simpletransformer_train(train_pairs, test_pairs):
    queries, rankings = train_pairs
    model_dir = "/tmp/simpletransformers_test/"
    train_args = {
        "max_seq_length": 512,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "save_eval_checkpoints": False,
        "train_batch_size": 64,
        "save_model_every_epoch": False,
        "best_model_dir": model_dir + "/best_model",
        "n_gpu": 2,
        "output_dir": model_dir,
        "use_multiprocessing": True,
        "evaluate_during_training": True,
        "evaluate_during_training_verbose": True,
        "logging_steps": 2000,
        "save_steps": 20000,
        "learning_rate": 6e-6,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "hidden_dropout_prob": 0.3,
        "vocab_size": 250000,
    }
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
