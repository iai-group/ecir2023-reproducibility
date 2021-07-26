"""Abstract query rewriting interface."""

from abc import ABC, abstractmethod

from treccast.core.query import Query
from treccast.core.context import Context


class QueryRewriter(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def rewrite_query(self, query: Query, context: Context) -> Query:
        """Rewrites a query based on context.

        Args:
            query: Unanswered query (last user utterance) to be
                rewritten.
            context: Conversation context up to the unanswered query.

        Returns:
            Rewritten query.
        """
        raise NotImplementedError
