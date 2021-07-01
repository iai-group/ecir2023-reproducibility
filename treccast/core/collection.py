"""Interfaces for collections."""

from abc import ABC

from elasticsearch.client import Elasticsearch


class Collection(ABC):
    def __init__(self, index_name: str) -> None:
        """Initializes collection (abstract) superclass.

        Args:
            index_name (str): Index name.
        """
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        return self._index_name


class ElasticSearchIndex(Collection):
    def __init__(
        self, index_name: str, hostname: str = "localhost:9200"
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name (str): Index name.
            hostname (str, optional): Host name and port (defaults to
                "localhost:9200").
        """
        super().__init__(index_name)
        self._es = Elasticsearch(hostname)

    @property
    def es(self) -> Elasticsearch:
        return self._es
