"""Class to fine-tune T5 models for Query Rewriting using simpletransformers
library."""

import argparse
import logging
import os
from typing import Dict, List

import pandas as pd
import torch.multiprocessing
from datasets import DatasetDict, load_dataset, load_metric
from simpletransformers.t5 import T5Args, T5Model

# The default name of the pretrained base model to be fine-tuned.
_BASE_MODEL_NAME = "t5"
# The default type of the pretrained model to be fine-tuned.
_MODEL_TYPE = "t5-base"
# The default location of the dataset to be used for model fine-tuning.
_DATASET_QRECC = "data/fine_tuning/rewriter/qrecc/"
# The portion of the training dataset to be extracted for validation.
_VALIDATION_PORTION = 0.1
# Special token to separate input sequences.
_SEPARATOR = "<sep>"
# The default location for the fine-tuned model.
_MODEL_LOCATION = (
    "data/fine_tuning/rewriter/qrecc/T5_QReCC_st_WaterlooClarke-train/"
)
# Location of the best fine-tuned model.
_BEST_MODEL_LOCATION = "best_model/"
# Whether train set should be split to train and validation sets.
_SPLIT_TRAIN_DATASET = True
# Maximum input sequence length in terms.
_MAX_SEQ_LENGTH = 512
# Whether all previous canonical responses should be included in the input. If
# false only the last one is included.
_INCLUDE_PREVIOUS_RESPONSES = False


class SimpleTransformersRewriterFinetuning:
    def __init__(
        self,
        train_args: T5Args,
        dataset: str,
        base_model_name: str,
        model_type: str,
    ) -> None:
        """Instantiates an object to fine-tune a model for Query Rewriting task.

        Args:
            train_args: Dictionary of training arguments.
            dataset: Path to the dataset used for fine-tuning. It should contain
              train and test sets in train.json and test.json files
              respectively.
            base_model_name: Base model name.
            model_type: Specific model type.
        """

        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        data_files = {
            "train": dataset + "train.json",
            "test": dataset + "test.json",
        }
        self._dataset = load_dataset(
            "json", data_files=data_files, field="data"
        )

        self._cuda_available = torch.cuda.is_available()

        # Load a pretrained model
        self._model = T5Model(
            base_model_name,
            model_type,
            args=train_args,
            use_cuda=self._cuda_available,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
        )

    @staticmethod
    def construct_input_data(
        context: List[str],
        question: List[str],
        separator: str = _SEPARATOR,
        include_previous_responses: bool = _INCLUDE_PREVIOUS_RESPONSES,
    ) -> str:
        """Constructs input data for the model from the provided data sample.

        Args:
            context: List of rewritten questions and answers from previous
              turns.
            question: The original question from the current turn.
            separator: Special token to separate input sequences.
            include_previous_responses: Whether all previous canonical responses
              should be included in the input. If false only the last one is
              included.

        Returns:
            Context utterances merged with current question using a separator.
            If the input sequence is too long (> _MAX_SEQ_LENGTH) last canonical
            response is cut.
        """
        if not include_previous_responses and len(context) > 0:
            context = context[0::2] + [context[-1]]

        split_context = [c.split(" ") for c in context]
        split_question = question.split(" ")
        if (
            sum([len(c) for c in split_context]) + len(split_question)
            > _MAX_SEQ_LENGTH
        ):
            split_previous_questions = split_context[:-1]
            split_last_canonical_response = split_context[-1]

            all_questions_length = sum(
                [len(c) for c in split_previous_questions]
            ) + len(split_question)
            split_position = len(split_last_canonical_response) - (
                _MAX_SEQ_LENGTH - all_questions_length
            )
            cut_canonical_response = split_last_canonical_response[
                :(-split_position)
            ]
            return separator.join(
                context[:-1] + [" ".join(cut_canonical_response)] + [question]
            )
        else:
            return separator.join(context + [question])

    @staticmethod
    def construct_df_from_dataset(
        dataset: Dict[str, List[str]],
        separator: str = _SEPARATOR,
        include_previous_responses: bool = _INCLUDE_PREVIOUS_RESPONSES,
    ) -> pd.DataFrame:
        """Creates a Pandas DataFrame from the given dataset.

        Args:
            dataset: The dataset to be processed.
            separator: Special token to separate input sequences.
            include_previous_responses: Whether all previous canonical responses
              should be included in the input. If false only the last one is
              included.

        Returns:
            pd.DataFrame: Pandas DataFrame with prefix, input_text and
            target_text columns.
        """
        constructed_input_data = [
            SimpleTransformersRewriterFinetuning.construct_input_data(
                context,
                question,
                separator,
                include_previous_responses,
            )
            for context, question in zip(
                dataset["Context"], dataset["Question"]
            )
        ]

        constructed_target_data = [rewrite for rewrite in dataset["Rewrite"]]

        prefix = "rewrite_question"
        constructed_prefix = [prefix] * len(constructed_target_data)

        return pd.DataFrame(
            {
                "prefix": constructed_prefix,
                "input_text": constructed_input_data,
                "target_text": constructed_target_data,
            }
        )

    def partition_dataset(
        self,
        separator: str = _SEPARATOR,
        split_train_dataset: bool = _SPLIT_TRAIN_DATASET,
        include_previous_responses: bool = _INCLUDE_PREVIOUS_RESPONSES,
    ) -> List[pd.DataFrame]:
        """Splits dataset to training, validation and test partitions.

        Args:
            separator: Special token to separate input sequences.
            split_train_dataset: Flag to decide whether train set should be
              split to train and validation sets.
            include_previous_responses: Whether all previous canonical responses
              should be included in the input. If false only the last one is
              included.

        Returns:
            Dataset with training, validation and test partitions for model
            fine-tuning.
        """
        valid_dataset = None

        if split_train_dataset:
            train_val_dataset = self._dataset["train"].train_test_split(
                test_size=_VALIDATION_PORTION
            )
            self._dataset = DatasetDict(
                {
                    "train": train_val_dataset["train"],
                    "test": self._dataset["test"],
                    "valid": train_val_dataset["test"],
                }
            )

            valid_dataset = (
                SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                    self._dataset["valid"],
                    separator,
                    include_previous_responses,
                )
            )

        train_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["train"], separator, include_previous_responses
            )
        )

        test_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["test"], separator, include_previous_responses
            )
        )

        return train_dataset, valid_dataset, test_dataset

    @staticmethod
    def rouge_metric(true_labels: List[str], predictions: List[str]) -> float:
        """Calculates ROUGE_1 metric.

        Args:
            true_labels: True labels.
            predictions: Model predictions.

        Returns:
            Rouge1 score (unigram based scoring).
        """
        rouge = load_metric("rouge")
        return rouge.compute(predictions=predictions, references=true_labels)[
            "rouge1"
        ]

    @staticmethod
    def rouge_metric_fmeasure(
        true_labels: List[str], predictions: List[str]
    ) -> float:
        """Calculates ROUGE_1 metric (F-measure).

        Args:
            true_labels: True labels.
            predictions: Model predictions.

        Returns:
            F-measure of Rouge1 score (unigram based scoring).
        """
        return SimpleTransformersRewriterFinetuning.rouge_metric(
            true_labels=true_labels, predictions=predictions
        ).mid.fmeasure

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        """Fine-tunes the pretrained model.

        Args:
            train_df: Training dataset.
            valid_df: Validation dataset.
        """
        self._model.train_model(
            train_df,
            eval_data=valid_df,
            rouge=SimpleTransformersRewriterFinetuning.rouge_metric_fmeasure,
        )

    @staticmethod
    def get_simpletransformers_args(model_dir: str) -> T5Args:
        """Gets arguments for training the model.

        Args:
            model_dir: Path to store the trained model.

        Returns:
            T5Args with arguments specified for fine-tuning.
        """

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model_args = T5Args()
        model_args.max_seq_length = _MAX_SEQ_LENGTH
        model_args.num_train_epochs = 3
        model_args.evaluate_generated_text = True
        model_args.evaluate_during_training = True
        model_args.evaluate_during_training_verbose = True
        model_args.overwrite_output_dir = True
        model_args.best_model_dir = model_dir + _BEST_MODEL_LOCATION
        model_args.output_dir = model_dir
        model_args.cache_dir = model_dir + "cache_dir/"
        model_args.train_batch_size = 2
        model_args.eval_batch_size = 2
        model_args.use_multiprocessing = False
        model_args.n_gpu = 1
        model_args.save_steps = 10000
        model_args.evaluate_during_training_steps = 10000
        model_args.dataloader_num_workers = 0
        model_args.process_count = 1
        model_args.use_multiprocessing_for_evaluation = False
        model_args.fp16 = False
        model_args.learning_rate = 5e-5
        model_args.dataloader_num_workers = 0
        model_args.use_multiprocessed_decoding = False

        return model_args


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(
        prog="simpletransformers_rewriter_finetunig.py"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=_BASE_MODEL_NAME,
        help="The name of the base model to be used for fine-tuning",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=_MODEL_TYPE,
        help="Specific model type to be used for fine-tuning",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DATASET_QRECC,
        help="The path to the dataset to be used for fine-tuning",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=_MODEL_LOCATION,
        help="The output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--split_train_dataset",
        action="store_true",
        help="Whether train set should be split to train and validation sets.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=_SEPARATOR,
        help="Special token to separate input sequences.",
    )
    parser.add_argument(
        "--validation_portion",
        type=float,
        default=_VALIDATION_PORTION,
        help="The portion of training dataset to be extracted for validation.",
    )
    parser.add_argument(
        "--include_previous_responses",
        action="store_false",
        help="Whether all previous canonical responses should be included in\n"
        "the input. If false only the last one is included.",
    )
    return parser.parse_args()


def main(args):
    """Fine-tunes a base model.

    Args:
        args: Command line arguments.
    """

    model_args = (
        SimpleTransformersRewriterFinetuning.get_simpletransformers_args(
            args.model_dir
        )
    )

    st_rewriter_finetuning = SimpleTransformersRewriterFinetuning(
        train_args=model_args,
        dataset=args.dataset,
        base_model_name=args.base_model_name,
        model_type=args.model_type,
    )

    train_df, valid_df, test_df = st_rewriter_finetuning.partition_dataset(
        separator=args.separator,
        split_train_dataset=args.split_train_dataset,
        include_previous_responses=args.include_previous_responses,
    )

    if args.split_train_dataset:
        st_rewriter_finetuning.train(train_df, valid_df)
    else:
        st_rewriter_finetuning.train(train_df, test_df)
    print("*** Model trained ***")

    # Load the fine-tuned model from the model location
    fine_tuned_model = T5Model(
        args.base_model_name,
        args.model_dir + _BEST_MODEL_LOCATION,
        args=model_args,
        use_cuda=st_rewriter_finetuning._cuda_available,
    )

    # Evaluate the fine-tuned model
    print("*** Evaluate the model ***")
    fine_tuned_model.eval_model(
        test_df, rouge=SimpleTransformersRewriterFinetuning.rouge_metric
    )

    # Make sample predictions with the fine-tuned model
    to_predict = [
        "rewrite_question: What is a physician's assistant?<sep>physician\n"
        "assistants are medical providers who are licensed to diagnose and\n"
        "treat illness and disease and to prescribe medication for patients.\n"
        "<sep>What are the educational requirements required to become one?",
        "rewrite_question: When did the 21st edition of the Commonwealth"
        " Games\nstart<sep>The 2018 Commonwealth Games, officially known as the"
        " XXI\nCommonwealth Game were held on the Gold Coast, Queensland,"
        " Australia,\nbetween 4 and 15 April 2018.<sep>When were the original"
        " first games\nheld",
        "rewrite_question: Who played the original jason in friday the"
        " 13th\n<sep>Ari Lehman is the actor/singer who has the unique honor of"
        " having\nplayed the role of the First Jason Voorhees in the Paramount"
        " Classic\nHorror Film Friday the 13th\u201c<sep>Besides Ari Lehman"
        " what other\nactor played the role of jason in friday the"
        " 13th.<sep>Steve Dash was\nthe stuntman/actor who played Jason in"
        " nearly every scene of  Friday\nThe 13th part 2 the movie, except the"
        " unmasked jump scare at the end.\n<sep>Any other",
    ]
    preds = fine_tuned_model.predict(to_predict)

    print("*** Sample rewrites generated by the model ***")
    print(preds)


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
