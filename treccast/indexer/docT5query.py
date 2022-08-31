"""
Contains DocT5Query used for generating queries given a piece of text.

Common approaches:
 * For MS_MARCO v1 generate 40 queries with model trained on MS_MARCO v1
    Model: "castorini/doc2query-t5-base-msmarco"

 * For MS_MARCO v2 generate 20 queries (due to size) with model trained on
    MS_MARCO v2
    NB! This model is not publicly available!

In both cases top-k sampling is used where k=10. It was shown to have better
performance than beam search.
"""

from typing import List, Union

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

_DocT5Queries = List[str]
_BatchDocT5Queries = List[List[str]]

# TODO See if MS-MARCO v2 is/will be available
_DOC2QUERY_MODEL = "castorini/doc2query-t5-base-msmarco"
# Number of highest probability tokens to use for sampling
TOP_K = 10


class DocT5Query:
    def __init__(
        self, n_queries: int = 20, top_k: int = TOP_K, device: str = None
    ) -> None:
        """Initializes docT5query for predicting possible queries from text.

        Args:
            n_queries: how many queries to predict per document (defaults
              to 20).
            top_k (optional): Number of highest scoring tokens to consider for
              sampling. Defaults to 10.
            device (optional): device to use. Defaults to None.

        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = T5Tokenizer.from_pretrained(_DOC2QUERY_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(
            _DOC2QUERY_MODEL
        ).eval()
        self.model.to(self.device)

        self.n_queries = n_queries
        self.top_k = top_k

    def _generate(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Union[_DocT5Queries, _BatchDocT5Queries]:
        """Generates sequences given input_ids.

        Args:
            input_ids: Tokenized inputs to use for generation.

        Returns:
            Depending on the format of input_ids and **kwargs, returns a list of
            sequences or a list of lists of sequences.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            **kwargs,
            max_length=64,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.n_queries,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_queries(self, doc_text: str) -> _DocT5Queries:
        """Generates n_queries for a given document text.

        Args:
            doc_text: Document text.

        Returns:
            List of len self.n_queries where each element is a predicted query
              for given doc_text.
        """

        input_ids = self.tokenizer.encode(doc_text, return_tensors="pt").to(
            self.device
        )

        return self._generate(input_ids)

    def batch_generate_queries(
        self, doc_texts: List[str]
    ) -> _BatchDocT5Queries:
        """Generates n_queries for each document text in doc_texts.

        Args:
            doc_texts: List of document text.

        Returns:
            List of len self.n_queries where each element is a predicted query
              for given doc_text.
        """

        inputs = self.tokenizer.batch_encode_plus(
            doc_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        queries = self._generate(**inputs)
        return [
            queries[self.n_queries * i : self.n_queries * (i + 1)]
            for i in range(len(doc_texts))
        ]
