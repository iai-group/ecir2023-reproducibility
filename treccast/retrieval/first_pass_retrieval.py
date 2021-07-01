"""Interface for first pass retrieval."""

from typing import List

from treccast.core.collection import Collection
from treccast.core.query.query import Query
from treccast.core.ranking import Ranking


class FirstPassRetrieval:
    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    def retrieve(self, query: Query) -> Ranking:
        pass

    def batch_retrieve(self, queries: List[Query]) -> List[Ranking]:
        return [self.retrieve(query) for query in queries]
