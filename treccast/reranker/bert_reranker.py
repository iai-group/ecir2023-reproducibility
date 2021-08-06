from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from treccast.reranker.reranker import NeuralReranker, validate, Batch


class BERTReranker(NeuralReranker):
    def __init__(
        self,
        model_name: str = "nboost/pt-bert-base-uncased-msmarco",
        max_seq_len: int = 256,
    ) -> None:
        """BERT reranker. Currently only supports BERT type architecture.

        Args:
            model_name (optional): Location to the model. Defaults to
                "nboost/pt-bert-base-uncased-msmarco".
            max_seq_len (optional): Maximal number of tokens. Defaults
                to 256.
        """
        super().__init__(max_seq_len)

        self._tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir="data/models", use_fast=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir="data/models"
        )

        self._model.to(self._device, non_blocking=True)

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
        # TODO Split into manageable batch sizes.
        # https://github.com/iai-group/trec-cast-2021/issues/66
        input_ids, attention_mask, token_type_ids = self._encode(
            query, documents
        )

        with torch.no_grad():
            logits = self._model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0]

            return logits.tolist()

    def _encode(self, query: str, documents: List[str]) -> Batch:
        """Tokenize and collate a number of single inputs, adding special
        tokens and padding.

        Returns:
            Batch: Input IDs, attention masks, token type IDs
        """
        inputs = self._tokenizer.batch_encode_plus(
            [[query, document] for document in documents],
            add_special_tokens=True,
            return_token_type_ids=True,
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
        token_type_ids = torch.tensor(inputs["token_type_ids"]).to(
            self._device, non_blocking=True
        )

        return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    ranker = BERTReranker()
    validate(ranker)
