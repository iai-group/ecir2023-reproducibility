"""Implements query with string literal of query utterance.
"""

from treccast.core.query.query import Query


class LiteralQuery(Query):
    def __init__(self, query_id: str, question: str) -> None:
        """Initializes a literal query from a question string.

        Args:
            query_id: Query ID.
            question: Question (e.g., manually re-written user utterance).
        """
        super().__init__(query_id, question)
