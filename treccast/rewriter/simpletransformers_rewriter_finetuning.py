"""Class to fine-tune T5 models for Query Rewriting using simpletransformers
library."""

import logging
from typing import Dict, List

import pandas as pd
from datasets import DatasetDict, load_dataset

# The default location of the dataset to be used for model fine-tuning.
_DATASET_QRECC = "data/fine_tuning/rewriter/qrecc/"
# The portion of the training dataset to be extracted for validation.
_VALIDATION_PORTION = 0.1
# Special token to separate input sequences.
_SEPARATOR = "<sep>"


class SimpleTransformersRewriterFinetuning:
    def __init__(self, dataset: str) -> None:
        """Instantiates an object to fine-tune a model for Query Rewriting task.

        Args:
            dataset: Path to the dataset used for fine-tuning. It should contain
            train and test sets in train.json and test.json files
            respectively.
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
        self, separator: str = _SEPARATOR
    ) -> List[pd.DataFrame]:
        """Splits dataset to training, validation and test partitions.

        Args:
            separator: Special token to separate input sequences.

        Returns:
            Dataset with training, validation and test partitions for model
            fine-tuning.
        """
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

        train_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["train"], separator
            )
        )

        valid_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["valid"], separator
            )
        )

        test_dataset = (
            SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
                self._dataset["test"], separator
            )
        )

        return train_dataset, valid_dataset, test_dataset


def main():
    """Fine-tunes a base model."""

    # TODO(lajewska): parsing of args will follow in a future PR
    st_rewriter_finetuning = SimpleTransformersRewriterFinetuning(
        dataset=_DATASET_QRECC,
    )

    train_df, valid_df, test_df = st_rewriter_finetuning.partition_dataset()

    print(train_df)
    print(valid_df)
    print(test_df)


if __name__ == "__main__":
    main()
