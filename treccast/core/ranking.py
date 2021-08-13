"""Represents a ranked list of items."""

import io
from typing import Dict, List, Tuple


class Ranking:
    def __init__(self, query_id: str, scored_docs: List[Dict] = None) -> None:
        """Instantiates a Ranking object using the query_id and a list of scored
        documents.

        Documents are stored unordered; sorting is done when fetching them.

        Args:
            query_id: Unique id for the query.
            scored_docs: List of dictionaries, where the keys `doc_id` and
                `score` are mandatory, and `content` is optional.
        """
        self._query_id = query_id
        self._scored_docs = scored_docs or []

    def __len__(self):
        return len(self._scored_docs)

    @property
    def query_id(self) -> str:
        return self._query_id

    def documents(self) -> Tuple[List[str], List[str]]:
        """Returns documents and their contents.

        Returns:
            Two parallel lists, containing document IDs and their content.
        """
        return (
            [doc["doc_id"] for doc in self._scored_docs],
            [doc.get("content") for doc in self._scored_docs],
        )

    def add_doc(
        self, doc_id: str, score: float, doc_content: str = None
    ) -> None:
        """Adds a new document to the ranking.

        Note: it doesn't check whether the document is already present.

        Args:
            doc_id: Document ID.
            score: The relevance score of the doc.
            doc_content (optional): String content of the document.
        """
        self._scored_docs.append(
            {"doc_id": doc_id, "score": score, "content": doc_content}
        )

    def add_docs(self, docs: List[Dict]) -> None:
        """Adds multiple documents to the ranking.

        Note: it doesn't check whether the document is already present.

        Args:
            docs: List of dictionaries, where the keys `doc_id` and
                `score` are mandatory, and `content` is optional.
        """
        self._scored_docs.extend(docs)

    def fetch_topk_docs(self, k: int = 1000) -> List[Dict]:
        """Fetches the top-k docs based on their score.

            If k > len(self._scored_docs), the slicing automatically
            returns all elements in the list in sorted order.
            Returns an empty array if there are no documents in the ranking.

        Args:
            k: Number of docs to fetch.

        Returns:
            Ordered list of dictionaries with doc_id, score, and (optional)
                content fields.
        """
        return sorted(
            self._scored_docs, key=lambda i: i["score"], reverse=True
        )[:k]

    def write_to_tsv_file(self, writer, query: str, k: int = 1000) -> None:
        """Writes the results of ranking to a tsv file in the format:
            query_id, query, passage_id, passage

        Args:
            writer: CSV writer that writes the tsv file.
                It should have delimiter="\t", and a header should be written
                before passing the writer to this function if required.
            query: The query/question content for which the passages are retrieved.
            k (optional): The number of documents to retrieve. Defaults to 1000.
        """
        for doc in self.fetch_topk_docs(k):
            writer.writerow(
                [self.query_id, query, doc["doc_id"], doc["content"]]
            )

    def write_to_trec_file(
        self, f_out: io.StringIO, run_id: str = "Undefined", k: int = 1000
    ) -> None:
        """Writes the top-k documents into an output TREC runfile.

        Args:
            f_out: Text file object open for writing.
            run_id (optional): Run ID. Defaults to "Undefined".
            k (optional): Number of documents to output. Defaults to 1000.
        """
        for rank, doc in enumerate(self.fetch_topk_docs(k)):
            f_out.write(
                " ".join(
                    [
                        self._query_id,
                        "Q0",
                        doc["doc_id"],
                        str(rank + 1),
                        str(doc["score"]),
                        run_id,
                    ]
                )
                + "\n"
            )
