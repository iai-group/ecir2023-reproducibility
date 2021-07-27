"""Represents a ranked list of items."""

from typing import Dict, List, Tuple, Optional


class Ranking:
    def __init__(
        self, query_id: str, scored_docs: Dict[str, Tuple[str, float]] = None
    ) -> None:
        """Instantiates a Ranking object using the query_id and a doc_id and
            score dictionary.

        Args:
            query_id: Unique id for the query.
            scored_docs: List of tuples of doc
                ids and score pairs.
        """
        # TODO change the Tuple[str, float] in scored_docs to a
        #  {"content": "content str", "score": 10.2}
        # issue https://github.com/iai-group/trec-cast-2021/issues/22
        self._query_id = query_id
        self._scored_docs = scored_docs if scored_docs is not None else {}

    def add_docs(
        self, scored_docs_list: List[Tuple[str, Tuple[str, float]]]
    ) -> None:
        """Adds a list of doc_id, score tuples to the dictionary.
            It updates existing entries.

        Args:
            scored_docs_list: List of doc_id,
                score tuples.
        """
        self._scored_docs.update(dict(scored_docs_list))

    def add_doc(self, doc_id: str, doc_content: str, score: float) -> None:
        """Adds a new score of a doc, given the doc_id.

        Args:
            doc_id (str): Unique id for the doc.
            doc_content (str): String content of the document.
            score (float): The relevance score of the doc.
        """
        self._scored_docs[doc_id] = (doc_content, score)

    def set_doc_score(self, doc_id: str, score: float) -> None:
        """Updates an existing score of a doc, given the doc_id.

        Args:
            doc_id: Unique id for the doc.
            score: The relevance score of the doc.
        """
        if doc_id in self._scored_docs:
            self._scored_docs[doc_id] = (self._scored_docs[doc_id][0], score)

    def get_doc_score(self, doc_id: str) -> Optional[float]:
        """Returns the score of the given doc id.

        Args:
            doc_id: Unique id for the doc.

        Returns:
            The relevance score of the doc.
        """
        return (
            self._scored_docs[doc_id][1]
            if doc_id in self._scored_docs
            else None
        )

    def fetch_topk_docs(self, k: int = 1000) -> List[Tuple[str, float]]:
        """Fetches the top k docs based on their score.

            If k > len(self._scored_docs), the slicing automatically
            returns all elements in the list in sorted order.
            Returns an empty array if there are no documents added to the
            ranker.

        Args:
            k: Number of docs to fetch.

        Returns:
            Ordered list of doc_id, score tuples.
        """
        return sorted(
            self._scored_docs.items(), key=lambda x: x[1][1], reverse=True
        )[:k]
