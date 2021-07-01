"""Interface for first pass retrieval."""

from abc import ABC, abstractmethod
from typing import List

from treccast.core.collection import Collection
from treccast.core.query.query import Query
from treccast.core.ranking import Ranking


class FirstPassRetrieval(ABC):
    def __init__(self, collection: Collection) -> None:
        """Abstract class for first-pass retrieval.

        Args:
            collection (Collection): Document collection.
        """
        self._collection = collection

    @abstractmethod
    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Interface for first-pass retrieval that needs to be implemented.

        Args:
            query (Query): Query instance.
            num_results (int, optional): Number of results to return (defaults
                to 1000).

        Raises:
            NotImplementedError: Raised if the method is not overwritten.

        Returns:
            Ranking: Ranking of documents.
        """
        raise NotImplementedError

    def batch_retrieve(self, queries: List[Query]) -> List[Ranking]:
        """Performs batch retrieval for a list of queries.

        Args:
            queries (List[Query]): List of input queries.

        Returns:
            List[Ranking]: List of rankings corresponding to the list of
                input queries.
        """
        return [self.retrieve(query) for query in queries]
