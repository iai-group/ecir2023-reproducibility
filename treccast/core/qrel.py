"""Represents a Qrel, a of set of relevance-judged items for a given query."""

from __future__ import annotations

import csv
from typing import Dict, List
from collections import defaultdict

from treccast.core.util.passage_loader import PassageLoader


def binarize_relevance(rel: float, threshold: int = 2) -> int:
    """Turns 0-4 relevance label into a binary relevance label (0/1).

    Args:
        rel: Relevance label between 0.0 through 4.0.
        threshold: Value at which to set the relevance to 1.

    Returns:
        Either 0 or 1.
    """
    return 1 if rel >= threshold else 0


class Qrel:
    def __init__(self, query_id: str) -> None:
        """Instantiates a Qrel object using the query_id and a list of judged
        documents.

        Documents are stored unordered with each relevance level.

        Args:
            query_id: Unique ID for the query.
        """
        self._query_id = query_id
        self._judged_docs = defaultdict(list)

    def __len__(self):
        return len(self._judged_docs)

    @property
    def query_id(self) -> str:
        return self._query_id

    def documents(self) -> Dict[str, str]:
        """Returns documents and their contents.

        Returns:
            Dictionary with doc_id as key and content as value.
        """
        return {
            doc["doc_id"]: doc.get("content")
            for docs in self._judged_docs.values()
            for doc in docs
        }

    def add_doc(self, doc_id: str, rel: int, doc_content: str = None) -> None:
        """Adds a new document to the Qrel.

        Note: it doesn't check whether the document is already present.

        Args:
            doc_id: Document ID.
            rel: The relevance label of the doc.
            doc_content (optional): String content of the document.
        """
        self._judged_docs[rel].append(
            {"doc_id": doc_id, "rel": rel, "content": doc_content}
        )

    def get_docs(self, rel: int = None) -> List[Dict]:
        """Fetches the docs with specified relevance label.

        Args:
            rel: Level of relevance to fetch docs from.

        Returns:
            Unordered list of dictionaries with doc_id, score, and (optional)
                content fields.
        """
        if rel:
            return self._judged_docs[rel]
        return [doc for docs in self._judged_docs.values() for doc in docs]

    @staticmethod
    def load_qrels_from_file(
        filepath: str, ploader: PassageLoader = None
    ) -> Dict[Qrel]:
        """Loads Qrels from TREC qrels file.

        Args:
            filepath: Path to TREC reqls file.
            ploader: PassageLoader that can retrieve passage content.

        Returns:
            Dictionary of Qrel objects with query ID as key.
        """
        qrels = {}
        with open(filepath, "r") as f_in:
            for q_id, _, doc_id, rel in csv.reader(f_in, delimiter=" "):
                rel = binarize_relevance(int(rel))
                if q_id not in qrels:
                    qrels[q_id] = Qrel(query_id=q_id)
                passage = ploader.get(doc_id=doc_id) if ploader else None
                qrels[q_id].add_doc(doc_id, rel, passage)
        return qrels
