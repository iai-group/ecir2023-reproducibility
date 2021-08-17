"""Load passage/question strings using Document/Query ID, instantiate objects.
"""
import csv

from collections import defaultdict
from typing import List

from treccast.core.collection import ElasticSearchIndex
from treccast.core.query.query import Query
from treccast.core.ranking import Ranking
from treccast.core.topic import construct_topics_from_file
from treccast.core.util.file_parser import FileParser


class PassageLoader(object):
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


class QueryLoader(object):
    def __init__(
        self,
        filepaths: List[str] = None,
        utterance_types: List[str] = ["manual", "raw"],
    ) -> None:
        """Loads queries to memory, provide single query or all as ordered list.

        Args:
            filepaths: All the topic .json filepaths to read queries from.
            utterance_type (optional): The variants of utterance to load, in
                sequence of preference. If the first type of utterance is
                available, use that, otherwise the next type, and so on. If no
                utterance of specified types is available, print warning.
                Defaults to ["manual", "raw"]. Type "automatic" is also
                acceptable.
        """
        self._utterance_types = utterance_types
        self._filepaths = filepaths
        if self._filepaths is None:
            self._filepaths = [
                # "data/topics/2019/ TODO .json",
                # TODO add on main: "data/topics/2020/manual_evaluation_topics_v1.0.json"
                # TODO remove on main:
                "data/topics/2020/2020_manual_evaluation_topics_v1.0.json"
                # "data/topics/2021/2021_manual_evaluation_topics_v1.0.json",
            ]
        self._topics = []
        self._queries = []
        self._query_dict = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Loads queries from every turn in every topic in the topics files."""
        for fp in self._filepaths:
            self._topics += construct_topics_from_file(fp)
        skips = 0
        for topic in self._topics:
            for turn in topic.turns:
                temp_query_id = "_".join(
                    [str(topic.topic_id), str(turn.turn_id)]
                )
                # Try to get an utterance of the most preferred type.
                temp_utterance = None
                for utterance_type in self._utterance_types:
                    if temp_utterance is None:
                        temp_utterance, _ = topic.get_question_and_context(
                            turn.turn_id, utterance_type
                        )
                # Skip if the utterance is not available as a specificed type.
                if temp_utterance is None:
                    print(
                        "Warning: None of the utterance types "
                        + f"{self._utterance_types} were available for query "
                        + f"{temp_query_id}."
                    )
                    skips += 1
                    continue
                temp_literal_query = Query(temp_query_id, temp_utterance)
                if temp_query_id in self._query_dict:
                    print("Duplicate query ID loaded.")
                    print(temp_query_id)
                assert temp_literal_query is not None
                self._query_dict[temp_query_id] = temp_literal_query
                # self._queries.append(temp_literal_query)
        print(
            f"Skipped {skips} query IDs due to no available utterance of "
            + "specified types."
        )

    def get(self, query_id) -> Query:
        return self._query_dict.get(query_id)

    @property
    def keys(self) -> List[str]:
        return list(self._query_dict.keys())


class QrelsLoader(object):
    def __init__(
        self,
        filepaths: List[str] = None,
        ploader: PassageLoader = None,
        qloader: QueryLoader = None,
    ) -> None:
        """Loads 'rankings' from QRELS files.
        Args:
            filepaths (optional): Paths to QRELS files. Defaults to None.
        """
        super().__init__()
        self._filepaths = filepaths
        self._qrels = {}
        if self._filepaths is None:
            self._filepaths = ["data/qrels/2019.txt", "data/qrels/2020.txt"]
        if ploader is None:
            self._ploader = PassageLoader()
        else:
            self._ploader = ploader
        if qloader is None:
            self._qloader = QueryLoader()
        else:
            self._qloader = qloader
        self._load_from_file()

    def _load_from_file(self):
        """Loads query utterances and ranked passages with relevance judgments."""
        for fp in self._filepaths:
            lines = FileParser.parse(fp)
            for i, line in enumerate(lines):
                q_id, _, doc_id, rel = line.split(" ")
                query = self._qloader.get(q_id)
                if query is None:
                    print(
                        f"Query not instantiated {q_id} referenced in QRELS fil"
                        + f"e {fp} on line {i+1}."
                    )
                    continue
                query_utterance = query.question
                if query_utterance is None:
                    print(f"Query missing utterance: {q_id}, {doc_id}, {rel}")
                if q_id not in self._qrels:
                    self._qrels[q_id] = Ranking(query_id=q_id)
                self._qrels[q_id].add_doc(
                    doc_id, float(rel), self._ploader.get(doc_id=doc_id)
                )

    def write_to_file(self, filepath: str) -> None:
        print(
            "Writing query utterances, passage content, and ranking scores to \
                file."
        )
        counter = defaultdict(int)
        with open(filepath, "w") as f_out:
            tsv_writer = csv.writer(f_out, delimiter="\t")
            for query_id, ranking in self._qrels.items():
                query = self._qloader.get(query_id=query_id)
                for doc in ranking.fetch_topk_docs(k=None):
                    query_utterance = query.question
                    passage = doc["content"]
                    score = str(float(doc["score"]))
                    counter[score] += 1
                    tsv_writer.writerow([query_utterance, passage, score])
        print("The scores occurred in the following frequencies:")
        for score, count in counter.items():
            print(f"{score}: {count}")

    def get_ranking(self, query_id: str) -> Ranking:
        ranking = self._qrels.get(query_id)
        if type(ranking) == Ranking:
            return ranking
        else:
            return None


class RunfileLoader:
    def __init__(
        self,
        filepath: str = None,
        ploader: PassageLoader = None,
        qloader: QueryLoader = None,
    ) -> None:
        """Loads rankings from queries and passages in runfiles.

        Args:
            filepath: Path to runfile.
        """
        super().__init__()
        self._filepath = filepath
        if self._filepath is None:
            self._filepath = (
                "data/runs/2020/org_baselines/y2_automatic_results_500.v1.0.run"
            )
        self._rankings = {}
        if ploader is None:
            self._ploader = PassageLoader()
        else:
            self._ploader = ploader
        if qloader is None:
            self._qloader = QueryLoader()
        else:
            self._qloader = qloader
        self._load_from_file()

    def _load_from_file(self):
        """Loads queries and rankings from runfile(s)."""
        with open(self._filepath, "r") as f_in:
            for i, line in enumerate(f_in):
                q_id, _, doc_id, rank, score, _ = line.split(" ")
                query = self._qloader.get(q_id)
                assert query is not None
                if q_id not in self._rankings:
                    self._rankings[q_id] = Ranking(query_id=q_id)
                self._rankings[q_id].add_doc(
                    doc_id,
                    float(score),
                    self._ploader.get(doc_id=doc_id),
                )

    def get_query(self, query_id: str) -> Query:
        """Get the query for a particular query ID.

        Args:
            query_id: Identifier for query ID.

        Returns:
            The corresponding query.
        """
        query = self._qloader.get(query_id)
        if type(query) == Query and query.question is not None:
            return query
        else:
            return None

    def get_ranking(self, query_id: str) -> Ranking:
        """Get the ranking for a particular query ID.

        Args:
            query_id: Identifier for query ID.

        Returns:
            The corresponding Ranking of passages.
        """
        ranking = self._rankings.get(query_id)
        if type(ranking) == Ranking:
            return ranking
        else:
            return None


if __name__ == "__main__":
    """The QrelsLoader() object reads both the QRELS files for 2019 and 2020,
    and is instantiated along with default components:
      - PassageLoader(): assumes Elasticsearch on localhost:9204, looks at index
        'ms_marco_trec_car_clean'.
      - QueryLoader(): assumes 2020 manual v1.0 topic file.
    """
    print("Starting QRELS loader...")
    qrloader = QrelsLoader()
    # For each line in the QRELS file, query ID and passage ID are resolved, and
    # the query utterance, passage, and relevance are written to the output file.
    print("Writing to file as fine-tuning data...")
    qrloader.write_to_file("data/finetuning/finetune_003.tsv")
