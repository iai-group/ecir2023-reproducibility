from typing import List

import torch
from ftfy import fix_text
from transformers import AutoTokenizer, T5ForConditionalGeneration

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
            "t5-base", cache_dir="data/models"
        )
        self._model = T5ForConditionalGeneration.from_pretrained(
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
