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
            query (Query): Unanswered query (last user utterance) to be
                rewritten.
            context (Context): Conversation context up to the unanswered query.

        Returns:
            Query: Rewritten query.
        """
        raise NotImplementedError
