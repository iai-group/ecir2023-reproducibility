"""Represents a sparse (keyword) query."""

from treccast.core.query.query import Query


class SparseQuery(Query):
    def __init__(self, id: str, text: str) -> None:
        super().__init__(id, text)
