"""Interface for first pass retrieval."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from treccast.core.base import Query
from treccast.core.collection import Collection
from treccast.core.ranking import Ranking


class Retriever(ABC):
    def __init__(self, collection: Collection) -> None:
        """Abstract class for first-pass retrieval.

        Args:
            collection: Document collection.
        """
        self._collection = collection

    @abstractmethod
    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Interface for first-pass retrieval that needs to be implemented.

        Args:
            query: Query instance.
            num_results: Number of results to return (defaults
                to 1000).

        Raises:
            NotImplementedError: Raised if the method is not overwritten.

        Returns:
            Ranking of documents.
        """
        raise NotImplementedError

    def batch_retrieve(self, queries: List[Query]) -> List[Ranking]:
        """Performs batch retrieval for a list of queries.

        Args:
            queries: List of input queries.

        Returns:
            List of rankings corresponding to the list of input queries.
        """
        return [self.retrieve(query) for query in queries]


class CachedRetriever(Retriever):
    def __init__(self, path: str) -> None:
        """Loads and caches first-pass rankings from a TSV file.

        Args:
            path: path to file from which to load rankings.
        """
        queries, rankings = Ranking.load_rankings_from_tsv_file(path)
        self._rankings = rankings
        self._queries = queries

    def retrieve(
        self, query: Query, num_results: int = 1000
    ) -> Tuple[Query, Ranking]:
        """Returns cached ranking for a given query.

        Args:
            query: Query for which to retrieve rankings.
            num_results: Number of documents to fetch. This is not in use in
                this class at the moment.
                Due to backward compatibility, we do not load scores so we
                cannot rank results.

        Returns:
            Query-Ranking pair for a given query.
        """
        return self._queries.get(query.query_id, query), self._rankings.get(
            query.query_id, Ranking(query.query_id)
        )
