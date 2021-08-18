"""Retrieves passages from Elasticsearch instance using IDs."""


from typing import List

from treccast.core.collection import ElasticSearchIndex


class PassageLoader(object):
    def __init__(
        self,
        hostname: str = "localhost:9204",
        index: str = "ms_marco_trec_car_clean",
    ) -> None:
        """Loads passage content from an ElasticSearch index.

        Args:
            hostname: Name of host and port number of Elasticsearch service.
            index: Name of index/collection on Elasticsearch service.
        """
        self._index = index
        self._es = ElasticSearchIndex(self._index, hostname=hostname).es
        self._cache = dict()

    def get(self, doc_id: str) -> str:
        """Load the passage content based on doc_id.

        Args:
            doc_id: Identifier of document (passage) to retrieve.

        Returns:
            The content of the indexed passage.
        """
        if doc_id not in self._cache:
            self._cache[doc_id] = self._es.get(self._index, doc_id)["_source"][
                "body"
            ]
        return self._cache[doc_id]

    def mget(self, doc_ids: List[str]) -> List[str]:
        """Load multiple passages based on a list of document IDs.

        The specified passages which are not already cached are retrieved, then
        all specified passages are loaded.

        Args:
            doc_ids: All the document identifiers with which to load content.

        Returns:
            The contents of each of the indexed passages.
        """
        missing_doc_ids = [
            doc_id for doc_id in doc_ids if doc_id not in self._cache
        ]
        result_dicts = self._es.mget(
            index=self._index, body={"ids": missing_doc_ids}
        )["docs"]
        results = [result["_source"]["body"] for result in result_dicts]
        for doc_id, result in zip(doc_ids, results):
            self._cache[doc_id] = result
        return [self._cache[doc_id] for doc_id in doc_ids]
