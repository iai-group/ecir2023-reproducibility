"""Represents a sparse query as a sequence of terms."""

from treccast.core.query.preprocessing.tokenizer import Tokenizer
from treccast.core.query.query import Query
from typing import List


class SparseQuery(Query):
    def __init__(
        self, query_id: str, question: str, tokenizer: Tokenizer = None
    ) -> None:
        """Initializes a sparse query from a question string.

        Args:
            query_id: Query ID.
            question: Question (raw user utterance).
            tokenizer: Tokenizer class that has get_tokens and preprocess_query
                methods.
        """
        super().__init__(query_id, question)
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def terms(self) -> List[str]:
        return self._tokenizer.get_tokens(self._question)

    @property
    def preprocessed_query(self) -> str:
        return self._tokenizer.preprocess_query(self._question)
