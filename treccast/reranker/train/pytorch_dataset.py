"""This class defines a class for point-wise ranking of query-document pairs."""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Iterable
from treccast.core.ranking import Ranking
from treccast.core.query import Query

Input = Tuple[str, str]
Batch = Tuple[
    torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor
]
TrainBatch = Tuple[Batch, torch.FloatTensor]
TrainingInput = Tuple[Input, List[int]]


class PointWiseDataset(Dataset):
    def __init__(
        self,
        queries: List[Query],
        rankings: List[Ranking],
        tokenizer: AutoTokenizer = None,
        max_len: int = 512,
    ) -> None:
        """Creates a pytorch dataset object for pointwise ranking.

        Args:
            queries: List of queries.
            rankings: List of ranking each one-to-one
                correpsonding to sparse queries.
            tokenizer: Bert tokenizer to tokenize the query,
                ranking pairs, if None is passed it creates a bert-base-cased
                tokenizer.
            max_len: Max length for the tokenizer to truncate the
                sequence.
        """
        self._query_doc_pairs = [
            (query, doc)
            for query, ranking in zip(queries, rankings)
            for doc in ranking.fetch_topk_docs()
        ]
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "bert-base-cased"
        )
        self._max_len = max_len

    def __getitem__(self, index: int):
        """Get a specific item in the dataset.

        Args:
            index: index of the item.

        Returns:
            Tuple of query, doc pair and corresponding score.
        """
        query, doc = self._query_doc_pairs[index]
        query_str = query.question
        doc_str = doc["content"]
        score = doc["score"]
        return (
            torch.tensor(int(query.query_id), dtype=torch.long),
            (query_str, doc_str),
            torch.tensor(score, dtype=torch.float),
        )

    def collate_bert(
        self, inputs: Iterable[Input], tokenizer: AutoTokenizer
    ) -> Batch:
        """Tokenize and collate a number of single BERT inputs, adding special
        tokens and padding.

        Args:
            inputs: The inputs
            tokenizer: Tokenizer
        Returns:
            Batch: Input IDs, attention masks, token type IDs
        """
        queries, docs = zip(*inputs)
        inputs = tokenizer(queries, docs, padding=True, truncation=True)
        if "token_type_ids" in inputs:
            return (
                torch.LongTensor(inputs["input_ids"]),
                torch.LongTensor(inputs["attention_mask"]),
                torch.LongTensor(inputs["token_type_ids"]),
            )
        else:
            return (
                torch.LongTensor(inputs["input_ids"]),
                torch.LongTensor(inputs["attention_mask"]),
            )

    def collate_fn(self, inputs: Iterable[TrainingInput]) -> TrainBatch:
        """Collate a number of pointwise inputs.
        Args:
            inputs: The inputs
        Returns:
            PointwiseTrainBatch: A batch of pointwise inputs
        """
        q_ids, inputs_, labels = zip(*inputs)
        return (
            torch.LongTensor(q_ids),
            self.collate_bert(inputs_, self._tokenizer),
            torch.FloatTensor(labels),
        )

    def __len__(self) -> int:
        """Dataset length.
        Returns:
            int: The number of training instances
        """
        return len(self._query_doc_pairs)
