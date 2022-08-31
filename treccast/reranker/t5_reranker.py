from collections import defaultdict
from itertools import permutations
from typing import List, Tuple

import torch
from ftfy import fix_text
from transformers import AutoTokenizer, T5ForConditionalGeneration
from treccast.core import NEURAL_MODEL_CACHE_DIR
from treccast.core.base import Query
from treccast.core.ranking import Ranking
from treccast.reranker.reranker import Batch, NeuralReranker


class T5Reranker(NeuralReranker):
    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        max_seq_len: int = 512,
        batch_size: int = 256,
    ) -> None:
        """T5 reranker.

        Args:
            model_name (optional): Location to the model. Defaults to
                "castorini/monot5-base-msmarco".
            max_seq_len (optional): Maximal number of tokens. Defaults
                to 512.
            batch_size (optional): Batch size. Defaults
                to 64.
        """
        super().__init__(max_seq_len, batch_size)

        self._tokenizer = AutoTokenizer.from_pretrained(
            "t5-base", cache_dir=NEURAL_MODEL_CACHE_DIR
        )
        self._model = (
            T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=NEURAL_MODEL_CACHE_DIR
            )
            .to(self._device, non_blocking=True)
            .eval()
        )

    def _get_logits(
        self, query: str, documents: List[str]
    ) -> List[List[float]]:
        """Returns logits from the neural model.

        Args:
            query: Query for which to evaluate.
            documents: List of documents to evaluate.

        Returns:
            A list containing two values for each document: the probability
                of the document being non-relevant [0] and relevant [1].
        """
        input_ids, attention_mask, decoder_input_ids = self._encode(
            query, documents
        )

        with torch.no_grad():
            all_tokens_logits = self._model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )[0][:, -1, :]
            # (batch size, vocabulary size)
            all_tokens_logits = all_tokens_logits

            # 6136, 1176 -> indexes of the tokens `false` and `true`
            # respectively.
            false_true_scores = all_tokens_logits[:, [6136, 1176]]
            log_scores = torch.nn.functional.log_softmax(
                false_true_scores, dim=1
            )
            return log_scores.tolist()

    def _encode(self, query: str, documents: List[str]) -> Batch:
        """Tokenize and collate a number of single inputs, adding special
        tokens and padding.

        Returns:
            Batch: Input IDs, attention masks, decoder IDs
        """
        inputs = self._tokenizer.batch_encode_plus(
            [
                fix_text(f"Query: {query} Document: {document} Relevant:")
                for document in documents
            ],
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self._max_seq_len,
        )

        input_ids = torch.tensor(inputs["input_ids"]).to(
            self._device, non_blocking=True
        )
        attention_mask = torch.tensor(inputs["attention_mask"]).to(
            self._device, non_blocking=True
        )

        decode_ids = torch.full(
            (input_ids.size(0), 1), self._model.config.decoder_start_token_id
        ).to(self._device, non_blocking=True)

        return input_ids, attention_mask, decode_ids


class DuoT5Reranker(NeuralReranker):
    def __init__(
        self,
        model_name: str = "castorini/duot5-base-msmarco",
        max_seq_len: int = 512,
        batch_size: int = 512,
    ) -> None:
        """Duo T5 reranker.

        Args:
            model_name (optional): Location to the model. Defaults to
              "castorini/duot5-base-msmarco".
            max_seq_len (optional): Maximal number of tokens. Defaults
              to 512.
            batch_size (optional): Batch size. Defaults
              to 64.
        """
        super().__init__(max_seq_len, batch_size)

        self._tokenizer = AutoTokenizer.from_pretrained(
            "t5-base", cache_dir=NEURAL_MODEL_CACHE_DIR
        )
        self._model = (
            T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=NEURAL_MODEL_CACHE_DIR
            )
            .to(self._device, non_blocking=True)
            .eval()
        )

    def rerank(self, query: Query, ranking: Ranking, top_k: int) -> Ranking:
        """Returns new ranking with updated scores from the neural reranker.

        Args:
            query: Query for which to re-rank.
            ranking: Current rankings for the query.
            top_k: Number of top documents in the ranking to be reranked.

        Returns:
            Ranking containing new scores for each document.
        """
        reranking = Ranking(ranking.query_id)
        reranking_top_k = ranking.fetch_topk_docs(top_k, unique=True)
        doc_ids = [doc["doc_id"] for doc in reranking_top_k]
        documents = [doc["content"] for doc in reranking_top_k]
        doc_pairs = list(permutations(documents, 2))
        doc_ids_pairs = list(permutations(doc_ids, 2))
        scores = defaultdict(float)
        for i in range(0, len(doc_pairs), self._batch_size):
            batch_doc_pairs = doc_pairs[i : i + self._batch_size]
            batch_doc_ids_pairs = doc_ids_pairs[i : i + self._batch_size]
            logits = self._get_logits(query.question, batch_doc_pairs)

            # Note: logit[0] corresponds to the document not being relevant and
            # logit[1] corresponds to the document being relevant.
            # This is the same for both BERT and T5 rerankers.
            for logit, doc_id in zip(logits, batch_doc_ids_pairs):
                scores[doc_id[0]] += logit[1]
                scores[doc_id[1]] += 1 - logit[1]

        reranking.add_docs(
            [
                {
                    "doc_id": doc["doc_id"],
                    "score": scores[doc["doc_id"]],
                    "content": doc["content"],
                }
                for doc in reranking_top_k
            ]
        )

        reranking.update(ranking._scored_docs)
        return reranking

    def _get_logits(
        self, query: str, documents: List[Tuple[str, str]]
    ) -> List[List[float]]:
        """Returns logits from the neural model.

        Args:
            query: Query for which to evaluate.
            documents: List of documents to evaluate.

        Returns:
            A list containing two values for each document: the probability
              of the document being non-relevant [0] and relevant [1].
        """
        input_ids, attention_mask, decoder_input_ids = self._encode(
            query, documents
        )

        with torch.no_grad():
            all_tokens_logits = self._model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )[0][:, -1, :]
            # (batch size, vocabulary size)
            all_tokens_logits = all_tokens_logits

            # 6136, 1176 -> indexes of the tokens `false` and `true`
            # respectively.
            false_true_scores = all_tokens_logits[:, [6136, 1176]]
            scores = torch.nn.functional.softmax(false_true_scores, dim=1)
            return scores.tolist()

    def _encode(self, query: str, documents: List[Tuple[str, str]]) -> Batch:
        """Tokenizes and collates a number of single inputs.

        It adds special tokens and padding.

        Args:
            query: Query to encode.
            documents: List of documents to encode.

        Returns:
            Batch: Input IDs, attention masks, decoder IDs.
        """
        inputs = self._tokenizer.batch_encode_plus(
            [
                fix_text(
                    f"Query: {query} Document0: {document[0]} "
                    f"Document1: {document[1]} Relevant:"
                )
                for document in documents
            ],
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self._max_seq_len,
        )

        input_ids = torch.tensor(inputs["input_ids"]).to(
            self._device, non_blocking=True
        )
        attention_mask = torch.tensor(inputs["attention_mask"]).to(
            self._device, non_blocking=True
        )

        decode_ids = torch.full(
            (input_ids.size(0), 1), self._model.config.decoder_start_token_id
        ).to(self._device, non_blocking=True)

        return input_ids, attention_mask, decode_ids
