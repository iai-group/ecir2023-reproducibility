"""Class to fine-tune T5 models for Query Rewriting using simpletransformers
library."""

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
        context: List[str], question: List[str], separator: str = _SEPARATOR
    ) -> str:
        """Constructs input data for the model from the provided data sample.

        Args:
            context: List of rewritten questions and answers from previous turns.
            question: The original question from the current turn.
            separator: Special token to separate input sequences.

        Returns:
            Context utterances merged with current question using a separator.
        """
        return separator.join(context + [question])

    @staticmethod
    def construct_df_from_dataset(
        dataset: Dict[str, List[str]], separator: str = _SEPARATOR
    ) -> pd.DataFrame:
        """Creates a Pandas DataFrame from the given dataset.

        Args:
            dataset: The dataset to be processed.
            separator: Special token to separate input sequences.

        Returns:
            pd.DataFrame: Pandas DataFrame with prefix, input_text and
            target_text columns.
        """
        constructed_input_data = [
            SimpleTransformersRewriterFinetuning.construct_input_data(
                context, question, separator
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
    ) -> List[pd.DataFrame]:
        """Splits dataset to training, validation and test partitions.

        Args:
            separator: Special token to separate input sequences.
            split_train_dataset: Flag to decide whether train set should be
              split to train and validation sets.

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
                    self._dataset["valid"], separator
                )
            )

        train_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["train"], separator
            )
        )

        test_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["test"], separator
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
        model_args.max_seq_length = 512
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

        return model_args


def main():
    """Fine-tunes a base model."""

    # TODO(lajewska): parsing of args will follow in a future PR
    model_args = (
        SimpleTransformersRewriterFinetuning.get_simpletransformers_args(
            _MODEL_LOCATION
        )
    )

    st_rewriter_finetuning = SimpleTransformersRewriterFinetuning(
        train_args=model_args,
        dataset=_DATASET_QRECC,
        base_model_name=_BASE_MODEL_NAME,
        model_type=_MODEL_TYPE,
    )

    train_df, valid_df, test_df = st_rewriter_finetuning.partition_dataset()

    print(train_df)
    print(valid_df)
    print(test_df)

    st_rewriter_finetuning.train(train_df, valid_df)
    print("*** Model trained ***")

    # Load the fine-tuned model from the model location
    fine_tuned_model = T5Model(
        _BASE_MODEL_NAME,
        _MODEL_LOCATION + _BEST_MODEL_LOCATION,
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
        "rewrite_question: When did the 21st edition of the Commonwealth Games\n"
        "start<sep>The 2018 Commonwealth Games, officially known as the XXI\n"
        "Commonwealth Game were held on the Gold Coast, Queensland, Australia,\n"
        "between 4 and 15 April 2018.<sep>When were the original first games\n"
        "held",
        "rewrite_question: Who played the original jason in friday the 13th\n"
        "<sep>Ari Lehman is the actor/singer who has the unique honor of having\n"
        "played the role of the First Jason Voorhees in the Paramount Classic\n"
        "Horror Film Friday the 13th\u201c<sep>Besides Ari Lehman what other\n"
        "actor played the role of jason in friday the 13th.<sep>Steve Dash was\n"
        "the stuntman/actor who played Jason in nearly every scene of  Friday\n"
        "The 13th part 2 the movie, except the unmasked jump scare at the end.\n"
        "<sep>Any other",
    ]
    preds = fine_tuned_model.predict(to_predict)

    print("*** Sample rewrites generated by the model ***")
    print(preds)


if __name__ == "__main__":
    main()
