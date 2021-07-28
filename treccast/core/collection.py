"""Interfaces for collections."""

from abc import ABC
from typing import Dict, Union

from elasticsearch.client import Elasticsearch


class Collection(ABC):
    def __init__(self, index_name: str) -> None:
        """Initializes collection (abstract) superclass.

        Args:
            index_name: Index name.
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
            index_name: Index name.
            hostname: Host name and port (defaults to
                "localhost:9200").
        """
        super().__init__(index_name)

        self._es = Elasticsearch(hostname)

    @property
    def es(self) -> Elasticsearch:
        return self._es

    def create_index(self) -> None:
        """Create new index if it does not exist."""
        if not self._es.indices.exists(self._index_name):
            self._es.indices.create(
                self._index_name, {"settings": self._get_default_settings()}
            )

    def delete_index(self) -> None:
        """Delete index if exists."""
        if self._es.indices.exists(self._index_name):
            self._es.indices.delete(self._index_name)

    def reset_index(self) -> None:
        """Deletes index if exists and creates a new one."""
        print(f"Resetting the index: {self.index_name}")
        self.delete_index()
        self.create_index()

    def update_similarity_parameters(self, **kwargs) -> None:
        """Updates similarity metric for an existing index with a custom
        configuration. Currently only works with BM25.
        """
        self._es.indices.close(self.index_name)
        self._es.indices.put_settings(
            {"index": self._get_BM25_similarity(**kwargs)}
        )
        self._es.indices.open(self.index_name)

    def _get_default_settings(self) -> Dict[str, Union[str, Dict]]:
        """Returns default index properties. This can be overridden with custom
        properties if needed.

        Returns:
            Dictionary with index properties.
        """
        return {"index": self._get_BM25_similarity()}

    def _get_BM25_similarity(
        self, b: float = 0.75, k1: float = 1.2
    ) -> Dict[str, Union[str, Dict]]:
        """Get dictionary containing settings for the default similarity with
        custom configuration.

        Args:
            b (optional): b parameter for BM25. Defaults to 0.75.
            k1 (optional): k1 parameter for BM25. Defaults to 1.2.

        Returns:
            Dictionary with index settings with BM25 similarity using custom
            parameters.
        """
        return {"similarity": {"default": {"type": "BM25", "b": b, "k1": k1}}}
