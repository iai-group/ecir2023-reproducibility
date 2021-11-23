"""Interface for first pass retrieval."""

from abc import ABC, abstractmethod
from typing import List

from treccast.core.collection import Collection
from treccast.core.query import Query
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
        self._rankings = Ranking.load_rankings_from_tsv_file(path)

    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Returns cached ranking for a given query.

        Args:
            query: Query for which to retrieve rankings.
            num_results: Number of documents to fetch. This is not in use in
                this class at the moment. see
                https://github.com/iai-group/trec-cast-2021/issues/228.
                Due to backward compatibility, we do not load scores so we cannot
                rank results.

        Returns:
            Ranking for a given query.
        """
        # FIXME https://github.com/iai-group/trec-cast-2021/issues/228
        # NB! Some old TSV files do not contain score so we cannot fetch top k
        # here. We need to re-run first-pass retrieval for relevant configs.
        return self._rankings.get(query.query_id, Ranking(query.query_id))
