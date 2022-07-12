"""Interfaces for collections."""

from abc import ABC
from typing import Any, Dict

import h5py
from elasticsearch.client import Elasticsearch
from h5py._hl.dataset import Dataset
from h5py._hl.files import File


class Collection(ABC):
    def __init__(self) -> None:
        """Initializes collection (abstract) superclass."""
        pass


class ElasticSearchIndex(Collection):
    def __init__(
        self, index_name: str, hostname: str = "localhost:9200", **kwargs
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name: Index name.
            hostname: Host name and port (defaults to
                "localhost:9200").
            **kwargs: Additional keyword arguments to be provided to the
                Elasticsearch instance.
        """
        super().__init__()
        self._index_name = index_name
        self._es = Elasticsearch(hostname, **kwargs)

    @property
    def es(self) -> Elasticsearch:
        return self._es

    @property
    def index_name(self) -> str:
        return self._index_name

    def create_index(self, use_analyzer: bool = True) -> None:
        """Create new index if it does not exist.


        Args:
            use_analyzer (optional): If True, use analyser to index and search
                documents. Defaults to True.
        """
        if not self._es.indices.exists(self._index_name):
            settings = self._get_default_settings()
            if use_analyzer:
                settings.update(self._get_analysis_settings())
            self._es.indices.create(self._index_name, {"settings": settings})
            print(
                "New Index: ",
                self._index_name,
                "\n",
                self._es.indices.get_settings()[self._index_name],
            )

    def delete_index(self) -> None:
        """Delete index if exists."""
        if self._es.indices.exists(self._index_name):
            self._es.indices.delete(self._index_name)

    def update_similarity_parameters(self, **kwargs) -> None:
        """Updates similarity metric for an existing index with a custom
        configuration. Currently only works with BM25.
        """
        self._es.indices.close(self._index_name)
        self._es.indices.put_settings(
            {"index": self._get_BM25_similarity(**kwargs)},
            index=self._index_name,
        )
        self._es.indices.open(self._index_name)

    def _get_default_settings(self) -> Dict[str, Any]:
        """Returns default index properties. This can be overridden with custom
        properties if needed.

        Returns:
            Dictionary with index properties.
        """
        return {"index": self._get_BM25_similarity()}

    def _get_BM25_similarity(
        self, b: float = 0.75, k1: float = 1.2
    ) -> Dict[str, Any]:
        """Get dictionary containing settings for the default similarity with
        custom configuration.

        Args:
            b (optional): b parameter for BM25. Defaults to 0.75.
            k1 (optional): k1 parameter for BM25. Defaults to 1.2.

        Returns:
            Dictionary with index settings with BM25 similarity using custom
            parameters.
        """
        return {
            "similarity": {"default": {"type": "BM25", "b": b, "k1": k1}},
            "max_result_window": 100000,
        }

    def _get_analysis_settings(self) -> Dict[str, Any]:
        """Elasticsearch analyzer. Should be overwritten if needed.

        Returns:
            Dictionary containing Elasticsearch configuration.
        """

        return {"analysis": {}}


class EmbeddingCollection(Collection):
    def __init__(self, filepath: str) -> None:
        """Initializes a collection of text embeddings.

        Args:
            filepath: File path to embeddings.
        """
        super().__init__()
        self._embeddings = self.load_embeddings(filepath)

    @property
    def embeddings(self) -> Dataset:
        """Dataset of embedding vectors."""
        return self._embeddings["embeddings"]

    @property
    def passage_ids(self) -> Dataset:
        """Dataset of passage ids."""
        return self._embeddings["passage_ids"]

    def load_embeddings(self, filepath: str) -> File:
        """Loads text embeddings.

        Args:
            filepath: File path to embeddings.

        Returns:
            File with embeddings and passage ids datasets.
        """
        if not filepath.endswith(".hdf5"):
            raise ValueError("File type not supported. Supported type: .hdf5")

        return h5py.File(filepath, "r")
