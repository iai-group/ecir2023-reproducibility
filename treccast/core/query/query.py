"""Abstract interface of a query."""

from abc import ABC


class Query(ABC):
    def __init__(self, query_id: str, question: str) -> None:
        """Abstract interface for a query.

        Args:
            query_id: Query ID.
            question: Question (raw user utterance).
        """
        self._query_id = query_id
        self._question = question

    @property
    def query_id(self) -> str:
        return self._query_id

    @property
    def question(self) -> str:
        return self._question
