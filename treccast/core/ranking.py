"""Represents a ranked list of items."""

from typing import Dict, List, Tuple, Optional


class Ranking:
    def __init__(
        self, query_id: str, scored_docs: Dict[str, float] = None
    ) -> None:
        """Instantiates a Ranking object using the query_id and a doc_id and
            score dictionary.

        Args:
            query_id (str): Unique id for the query.
            scored_docs (Dict[str, float]): List of tuples of doc
                ids and score pairs.
        """
        self._query_id = query_id
        self._scored_docs = scored_docs if scored_docs is not None else {}

    def add_docs(self, scored_docs_list: List[Tuple[str, float]]) -> None:
        """Adds a list of doc_id, score tuples to the dictionary.
            It updates existing entries.

        Args:
            scored_docs_list (List[Tuple[str, float]]): List of doc_id,
                score tuples.
        """
        self._scored_docs.update(dict(scored_docs_list))

    def set_doc_score(self, doc_id: str, score: float) -> None:
        """Adds a new or updates an existing score of a doc, given the doc_id.

        Args:
            doc_id (str): Unique id for the doc.
            score (float): The relevance score of the doc.
        """
        self._scored_docs[doc_id] = score

    def get_doc_score(self, doc_id: str) -> Optional[float]:
        """Returns the score of the given doc id.

        Args:
            doc_id (str): Unique id for the doc.

        Returns:
            float: The relevance score of the doc.
        """
        return (
            self._scored_docs[doc_id] if doc_id in self._scored_docs else None
        )

    def fetch_topk_docs(self, k: int) -> List[Tuple[str, float]]:
        """Fetches the top k docs based on their score.
            If k > len(self._scored_docs), the slicing automatically
            returns all elements in the list in sorted order.

        Args:
            k (int): Number of docs to fetch.

        Returns:
            List[Tuple[str, float]]: Ordered list of doc_id, score tuples.
        """
        return sorted(
            self._scored_docs.items(), key=lambda x: x[1], reverse=True
        )[:k]
