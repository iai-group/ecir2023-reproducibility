"""Load paragraph content using Document ID.
"""

from treccast.core.collection import ElasticSearchIndex
from typing import List


class ContentLoader(object):
    def __init__(
        self,
        hostname: str = "localhost:9204",
        index: str = "ms_marco_trec_car_clean",
    ) -> None:
        """Load passage content already indexed using Document ID.

        Args:
            hostname: Name of host and port number of Elasticsearch service.
            index: Name of index/collection on Elasticsearch service.
        """
        self._index = index
        self._es = ElasticSearchIndex(self._index, hostname=hostname).es
        self._cache = dict()

    def get(self, doc_id: str, recheck: bool = False) -> str:
        """Load the passage content based on doc_id.

        Args:
            doc_id: Identifier of document (passage) to retrieve.
            recheck (optional): Whether to retrieve and re-cache content.

        Returns:
            The content of the indexed passage.
        """
        if recheck or doc_id not in self._cache:
            passage = self._es.get(self._index, doc_id)["_source"]["body"]
            self._cache[doc_id] = passage
        else:
            passage = self._cache[doc_id]
        return passage

    def mget(self, doc_ids: List[str]) -> List[str]:
        """Load multiple passages based on a list of document IDs. 
        
        Adds new results to cache but always retrieves results from index, does
        not check cache first. 

        Args:
            doc_ids: All the document identifiers with which to load content.

        Returns:
            The contents of each of the indexed passages.
        """
        result_dicts = self._es.mget(index=self._index, body={"ids": doc_ids})[
            "docs"
        ]
        results = [result["_source"]["body"] for result in result_dicts]
        for doc_id, result in zip(doc_ids, results):
            self._cache[doc_id] = result
        return results


if __name__ == "__main__":
    loader = ContentLoader()
    # Example of getting some document.
    print(loader.get("CAR_3add84966af079ed84e8b2fc412ad1dc27800127"))
    print(
        loader.mget(
            [
                "CAR_3add84966af079ed84e8b2fc412ad1dc27800127",
                "CAR_5fa30140b395d7fead223e2bca8cc9b608bb51b4",
            ]
        )
    )
