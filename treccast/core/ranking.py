"""Represents a ranked list of items.
"""
from __future__ import annotations

import csv
import io
import sys
from typing import Dict, List, Tuple

from treccast.core.util.passage_loader import PassageLoader

# This is needed since some of the passages are too long.
csv.field_size_limit(sys.maxsize)


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

    def update(self, docs: List[Dict]) -> None:
        """Adds multiple documents to the ranking uniquely.

        Args:
            docs: List of dictionaries, where the keys `doc_id` and
                `score` are mandatory, and `content` is optional.
        """
        doc_ids, _ = self.documents()
        self._scored_docs.extend(
            [doc for doc in docs if doc["doc_id"] not in doc_ids]
        )

    def fetch_topk_docs(
        self, k: int = 1000, unique: bool = False
    ) -> List[Dict]:
        """Fetches the top-k docs based on their score.

            If k > len(self._scored_docs), the slicing automatically
            returns all elements in the list in sorted order.
            Returns an empty array if there are no documents in the ranking.

        Args:
            k: Number of docs to fetch.
            unique: If unique is True returns unique unique documents. In case
                of multiple documents with the same ID, returns the highest
                scoring. Defaults to False

        Returns:
            Ordered list of dictionaries with doc_id, score, and (optional)
                content fields.
        """
        sorted_docs = sorted(
            self._scored_docs, key=lambda i: i["score"], reverse=True
        )
        if unique:
            sorted_unique_docs = {}
            for doc in sorted_docs:
                if doc["doc_id"] not in sorted_unique_docs:
                    sorted_unique_docs[doc["doc_id"]] = doc
            sorted_docs = list(sorted_unique_docs.values())
        return sorted_docs[:k]

    def write_to_tsv_file(self, writer, query: str, k: int = 1000) -> None:
        """Writes the results of ranking to a tsv file in the format:
            query_id, query, passage_id, passage, score

        Note: The writer is of type csv._writer, but there is no simple way to
            access this type, the only possible solutions would be to add a new
            dependency or create a wrapper around the writer, see
            https://stackoverflow.com/q/51264355 for more information.

        Args:
            writer: CSV writer that writes the tsv file.
                It should have delimiter="\t", and a header should be written
                before passing the writer to this function if required.
            query: The query/question content for which the passages are retrieved.
            k (optional): The number of documents to retrieve. Defaults to 1000.
        """
        for doc in self.fetch_topk_docs(k, unique=True):
            writer.writerow(
                [
                    self.query_id,
                    query,
                    doc["doc_id"],
                    doc["content"],
                    doc["score"],
                ]
            )

    def write_to_trec_file(
        self,
        f_out: io.StringIO,
        run_id: str = "Undefined",
        k: int = 1000,
        remove_passage_id: bool = False,
    ) -> None:
        """Writes the top-k documents into an output TREC runfile.

        Args:
            f_out: Text file object open for writing.
            run_id (optional): Run ID. Defaults to "Undefined".
            k (optional): Number of documents to output. Defaults to 1000.
            remove_passage_id (optional): Removes passageID from the documentID
                (separated by "-"). Defaults to False.
        """
        doc_ids = set()
        for rank, doc in enumerate(self.fetch_topk_docs(k, unique=True)):
            # Leave out passageID from the docID.
            doc_id = (
                doc["doc_id"].split("-")[0]
                if remove_passage_id
                else doc["doc_id"]
            )
            if doc_id in doc_ids:  # Ignore duplicates
                continue

            f_out.write(
                " ".join(
                    [
                        self._query_id,
                        "Q0",
                        doc_id,
                        str(rank + 1),
                        str(doc["score"]),
                        run_id,
                    ]
                )
                + "\n"
            )
            doc_ids.add(doc_id)

    @staticmethod
    def load_rankings_from_runfile(
        filepath: str,
        ploader: PassageLoader = None,
    ) -> Dict[str, Ranking]:
        """Creates a list of Ranking objects from a TREC runfile.
        Also, optionally loads passage content.

        Args:
            filepath: Path to the runfile.
            ploader (optional): PassageLoader can retrieve passage contents from
            Elasticsearch instance. Defaults to None.

        Returns:
            A dictionary of the Ranking objects built from the runfile, with
            query ID as key..
        """
        rankings = {}
        with open(filepath, "r") as f_in:
            reader = csv.reader(f_in, delimiter=" ")
            for row in reader:
                q_id, _, doc_id, _, score, _ = row
                content = ploader.get(doc_id=doc_id) if ploader else None
                if q_id not in rankings:
                    rankings[q_id] = Ranking(query_id=q_id)
                rankings[q_id].add_doc(doc_id, float(score), content)
        return rankings

    @staticmethod
    def load_rankings_from_tsv_file(
        filepath: str,
    ) -> Dict[str, Ranking]:
        """Creates a list of Ranking objects from a TSV runfile.

        Args:
            filepath: Path to the runfile.

        Returns:
            A dictionary of the Ranking objects built from the runfile.
        """
        rankings = {}
        with open(filepath, "r") as f_in:
            reader = csv.reader(f_in, delimiter="\t")
            next(reader)
            for line in reader:
                q_id, _, doc_id, passage = line[:4]
                if q_id not in rankings:
                    rankings[q_id] = Ranking(query_id=q_id)
                rankings[q_id].add_doc(doc_id, 0, passage)
        return rankings
