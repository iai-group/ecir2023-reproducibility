"""Abstract representation of a query."""

from abc import ABC


class Query(ABC):
    def __init__(self, query_id: str) -> None:
        self._query_id = query_id

    @property
    def query_id(self) -> str:
        return self._query_id
