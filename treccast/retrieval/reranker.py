"""Abstract interface for a reranker."""

from abc import ABC
from typing import List

from treccast.core.query.query import Query
from treccast.core.ranking import Ranking


class Reranker(ABC):
    def __init__(self, rankings: List[Ranking], queries: List[Query]) -> None:
        self.__rankings = rankings
        self.__queries = queries

    def rerank(self) -> List[Ranking]:
        """Performs reranking.

        Returns:
            List of Ranking instances.
        """
        pass
