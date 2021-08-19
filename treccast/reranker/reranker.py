"""Abstract interface for a reranker."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from treccast.core.query import Query
from treccast.core.ranking import Ranking

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Reranker(ABC):
    def __init__(self) -> None:
        """Interface for a reranker."""
        pass

    @abstractmethod
    def rerank(self, query: Query, ranking: Ranking) -> Ranking:
        """Performs reranking.

        Returns:
            New Ranking instance with updated scores.
        """
        raise NotImplementedError


class NeuralReranker(Reranker, ABC):
    def __init__(
        self,
        max_seq_len: int = 512,
        batch_size: int = 8,
    ) -> None:
        """Neural reranker.

        Args:
            max_seq_len (optional): Maximal number of tokens. Defaults
                to 512.
            batch_size (optional): Batch size. Defaults
                to 8.
        """

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size

    def rerank(
        self,
        query: Query,
        ranking: Ranking,
    ) -> Ranking:
        """Returns new ranking with updated scores from the neural reranker.

        Args:
            query: Query for which to re-rank.
            ranking: Current rankings for the query.
            batch_size: Number of query-passage pairs per batch.

        Returns:
            Ranking containing new scores for each document.
        """
        reranking = Ranking(ranking.query_id)
        doc_ids, documents = ranking.documents()
        for i in range(0, len(documents), self._batch_size):
            batch_documents = documents[i : i + self._batch_size]
            batch_doc_ids = doc_ids[i : i + self._batch_size]
            logits = self._get_logits(query.question, batch_documents)

            # Note: logit[0] corresponds to the document not being relevant and
            # logit[1] corresponds to the document being relevant.
            # This is the same for both BERT and T5 rerankers.
            reranking.add_docs(
                [
                    {"doc_id": doc_id, "score": logit[1], "content": doc}
                    for (logit, doc_id, doc) in zip(
                        logits, batch_doc_ids, batch_documents
                    )
                ]
            )
        return reranking

    @abstractmethod
    def _get_logits(
        self, query: str, documents: List[str]
    ) -> List[List[float]]:
        """Returns logits from the neural model.

        Args:
            query: Query for which to evaluate.
            documents: List of documents to evaluate.

        Returns:
            List of lists containing two values for each document: the
                probability of the document being non-relevant [0] and
                relevant [1].
        """
        raise NotImplementedError
