"""Class to fine-tune BERT models using simpletransformers library."""

import argparse
from typing import Any, Dict, List

import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from treccast.core.base import Query
from treccast.core.ranking import Ranking
from treccast.core.util.fine_tuning.finetuning_data_loader import (
    FineTuningDataLoader,
)


class SimpleTransformersTrainer:
    def __init__(
        self,
        train_args: Dict[str, Any],
        base_model: str = "bert",
        bert_type: str = "nboost/pt-bert-base-uncased-msmarco",
    ) -> None:
        """Instantiates an object to train a Transformer model using the
        Simpletransformers module.

        Args:
            base_model: Base model type defaults to "bert".
            bert_type: Specific BERT type defaults to "".
            train_args: Dict of training args needed by Simpletransformers.
        """

        self._model = ClassificationModel(
            base_model,
            bert_type,
            num_labels=2,
            use_cuda=True,
            args=train_args,
        )

    @staticmethod
    def get_data_df(
        queries: List[Query], rankings: List[Ranking]
    ) -> pd.DataFrame:
        """Convert list of Query and Rankings into a Pandas dataframe for
        simpletransformer.

        Args:
            queries: List of queries.
            rankings: List of corresponding rankings.

        Returns:
            pd.DataFrame: Pandas dataframe with text_a, text_b and labels
            columns.
        """
        query_doc_pairs = [
            [query.question, doc.content, doc.score]
            for query, ranking in zip(queries, rankings)
            for doc in ranking.fetch_topk_docs()
        ]
        df = pd.DataFrame.from_records(
            query_doc_pairs, columns=["text_a", "text_b", "labels"]
        )
        return df

    def train(
        self,
        queries: List[Query],
        rankings: List[Ranking],
        val_size: float = 0.05,
    ) -> None:
        """Trains the simpletransformer model.

        Args:
            queries: List of Query objects.
            rankings:: List of Rankings corresponding to the Query objects.
            val_size: [description]. Validation split size (0 to 1).
            Defaults to 0.05.
        """
        data_df = SimpleTransformersTrainer.get_data_df(queries, rankings)
        # If validation size is specified create a validation split, otherwise
        # use full data as train split.
        if val_size > 0:
            train_df, val_df = train_test_split(data_df, test_size=val_size)
            self._model.train_model(train_df, eval_df=val_df)
        else:
            self._model.train_model(data_df)

    @staticmethod
    def get_default_simpletransformers_args(model_dir: str) -> Dict[str, Any]:
        """Returns default args for training simpletransformer model.

        Args:
            model_dir: Path to store the trained model.
        Returns:
            Dict[str, Any]: Dictionary of parameters.
        """
        train_args = {
            "max_seq_length": 512,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 10,
            "save_eval_checkpoints": False,
            "train_batch_size": 32,
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
        return train_args


def arg_parser() -> argparse.ArgumentParser:
    """Function to parse command line arguments.

    Returns: ArgumentParser object with parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset(s) to train on.",
        choices=["treccast", "wow", "both"],
        default="both",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--treccast_years",
        type=str,
        help=(
            "If treccast or both datasets are choosen in --dataset specific y1"
            " (2019) or y2 (2020) or both"
        ),
        choices=["y1", "y1y2"],
        default="y1y2",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--base_bert_type",
        type=str,
        help="Base BERT type",
        default="bert",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--bert_model_path",
        type=str,
        help="Base BERT type",
        default="nboost/pt-bert-base-uncased-msmarco",
        const=1,
        nargs="?",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    fnt_wow_loader = FineTuningDataLoader(
        file_name="data/fine_tuning/wizard_of_wikipedia/wow_finetune_train.tsv"
    )
    wow_queries, wow_rankings = fnt_wow_loader.get_query_ranking_pairs()
    if args.treccast_years == "y1":
        fnt_loader = FineTuningDataLoader(
            "data/fine_tuning/trec_cast/Y1_manual_qrels.tsv"
        )
    elif args.treccast_years == "y1y2":
        fnt_loader = FineTuningDataLoader(
            "data/fine_tuning/trec_cast/Y1Y2_manual_qrels.tsv"
        )
    trec_queries, trec_rankings = fnt_loader.get_query_ranking_pairs()

    if args.dataset == "treccast":
        queries, rankings = trec_queries, trec_rankings
    elif args.dataset == "wow":
        queries, rankings = wow_queries, wow_rankings
    elif args.dataset == "both":
        queries, rankings = trec_queries, trec_rankings
        queries.extend(wow_queries)
        rankings.extend(wow_rankings)
    else:
        raise ValueError("Either treccast, wow or both are supported.")
    model_name = args.bert_model_path.replace("/", "_")
    model_dir = f"data/models/finetuned_models/simpletransformers_{args.base_bert_type}_{model_name}_{args.dataset}_{args.treccast_years}/"  # noqa E501 long path
    print(model_dir)
    train_args = SimpleTransformersTrainer.get_default_simpletransformers_args(
        model_dir
    )
    st_trainer = SimpleTransformersTrainer(
        train_args=train_args,
        base_model=args.base_bert_type,
        bert_type=args.bert_model_path,
    )
    st_trainer.train(queries=queries, rankings=rankings)
