"""Topic class represents conversation as sequence of Turns each consisting
of a Question and Context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from treccast.core.base import Context, Document, Query
from treccast.core.util.passage_loader import PassageLoader


class QueryRewrite(Enum):
    AUTOMATIC = 1
    MANUAL = 2
    MIXED_INITIATIVE = 3


@dataclass
class Turn:
    turn_id: str
    canonical_result_id: str
    result_turn_dependence: int
    query_turn_dependence: list
    raw_utterance: str
    mi_expanded_utterance: str = None
    automatic_rewritten_utterance: str = None
    manual_rewritten_utterance: str = None
    passage_id: str = None
    response: str = None
    provenance: List[str] = None
    passage: str = None
    canonical_passage: str = None
    provenance_passages: List[str] = None
    turn_leaf_id: str = None

    def __post_init__(self):
        if self.passage_id is not None:
            self.canonical_result_id = (
                f"{self.canonical_result_id}-{self.passage_id}"
            )

    def get_utterance(self, query_rewrite: QueryRewrite = None) -> str:
        """Returns the text utterance for this turn, optionally, with query
        rewriting applied.

        Args:
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).

        Raises:
            ValueError: If the specified query rewrite is unavailable (this
              likely means that wrong topic file is used).

        Returns:
            Utterance string.
        """
        utterance = self.raw_utterance
        if query_rewrite == QueryRewrite.AUTOMATIC:
            utterance = self.automatic_rewritten_utterance
        elif query_rewrite == QueryRewrite.MANUAL:
            utterance = self.manual_rewritten_utterance
        elif query_rewrite == QueryRewrite.MIXED_INITIATIVE:
            utterance = self.mi_expanded_utterance

        if not utterance:
            raise ValueError(
                "Requested query rewrite is unavailable for this turn "
                f"(#{self.turn_id})"
            )

        return utterance

    def get_passage_content(self, use_answer_rewrite: bool = False) -> str:
        """Returns passage content.

        Args:
            use_answer_rewrite: If false, returns full passage, otherwise
              uses answer rewrite. Defaults to False.

        Returns:
            Passage content.
        """
        if use_answer_rewrite and self.passage:
            return self.passage
        return self.canonical_passage

    def get_provenance_content(
        self, use_answer_rewrite: bool = False
    ) -> List[str]:
        """Returns contents of all passages in provenance.

        Args:
            use_answer_rewrite: If false, returns list of passages, otherwise
              uses answer rewrite (response). Defaults to False.

        Returns:
            Passage content.
        """
        if use_answer_rewrite and self.response:
            return [self.response]
        return self.provenance_passages


@dataclass
class Topic:
    topic_id: int
    description: str
    title: str
    turns: List[Turn]

    def get_turn(self, turn_id: str) -> Turn:
        """Returns a given topic turn.

        Args:
            turn_id: Turn ID.

        Raises:
            IndexError: When turn ID is out of bounds.

        Returns:
            Turn instance.
        """
        if turn_id not in [turn.turn_id for turn in self.turns]:
            raise IndexError(f"Invalid turn_id: {turn_id}")
        return next(turn for turn in self.turns if turn.turn_id == turn_id)

    def get_query_id(self, turn_id: str) -> str:
        """Returns query ID corresponding to a given topic turn.

        Args:
            turn_id: Turn ID.

        Returns:
            Query ID.
        """
        return f"{self.topic_id}_{turn_id}"

    def get_query(
        self, turn_id: str, query_rewrite: QueryRewrite = None
    ) -> Query:
        """Returns query corresponding to a given turn, optionally, with query
        rewriting applied.

        Args:
            turn_id: Turn ID.
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).

        Returns:
            Query object.
        """
        utterance = self.get_turn(turn_id).get_utterance(query_rewrite)
        return Query(
            self.get_query_id(turn_id),
            utterance,
            self.get_turn(turn_id).turn_leaf_id,
        )

    def get_queries(self, query_rewrite: QueryRewrite = None) -> List[Query]:
        """Returns a list of queries corresponding to each of the turns,
        optionally, with query rewriting applied.

        Args:
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).

        Returns:
            List of Query objects.
        """
        return [
            self.get_query(turn.turn_id, query_rewrite) for turn in self.turns
        ]

    def get_contexts(
        self,
        year: str,
        query_rewrite: QueryRewrite = None,
        use_answer_rewrite: bool = False,
    ) -> List[Context]:
        """Gets a list of contexts for each turn.

        Args:
            year: Year (2019_train, 2019, 2020, 2021, or 2022).
            query_rewrite (optional): Query rewrite variant to include in
              context (auto/manual). Defaults to None (i.e., raw).
            use_answer_rewrite (optional): If false, uses full canonical passage
              text(s) in context, otherwise only rewritten response. Defaults
              to False.

        Returns:
            List of Contexts corresponding to every turn with canonical
            responses included.
        """
        queries = self.get_queries(query_rewrite)[:-1]
        if year == "2022":
            canonical_responses = [
                turn.get_provenance_content(use_answer_rewrite)
                for turn in self.turns
            ][:-1]
        else:
            canonical_responses = [
                turn.get_passage_content(use_answer_rewrite)
                for turn in self.turns
            ][:-1]
        contexts = [None]
        for query, canonical_response in zip(queries, canonical_responses):
            context = Context()
            context.history = (
                contexts[-1].history.copy() if len(contexts) > 1 else []
            )
            if year == "2022":
                context.history.append(
                    (
                        query,
                        [
                            Document(None, passage)
                            for passage in canonical_response
                        ],
                    )
                )
            else:
                context.history.append(
                    (query, [Document(None, canonical_response)])
                )
            contexts.append(context)
        return contexts

    @staticmethod
    def get_filepath(
        year: str, query_rewrite: QueryRewrite = None, use_extended: bool = True
    ) -> str:
        """Returns file path to topic file to be used based on year and query
        rewriting applied.

        Args:
            year: Year (2019_train, 2019, 2020, or 2021).
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).
            use_extended (optional): If true, uses extended topic file version.

        Returns:
            Topic file path (relative to repo root).
        """
        filepath = f"data/topics/{year}/"
        if year == "2019_train":
            filepath += "train_topics"
        else:
            if query_rewrite == QueryRewrite.AUTOMATIC:
                variant = "automatic"
            elif query_rewrite == QueryRewrite.MIXED_INITIATIVE:
                variant = "mi"
            else:
                variant = "manual"
            filepath += f"{year}_{variant}_evaluation_topics"
        extend = "_extended" if use_extended else ""
        filepath += f"_v1.0{extend}.json"

        return filepath

    @staticmethod
    def load_topics_from_file(
        year: str, query_rewrite: QueryRewrite = None, use_extended: bool = True
    ) -> List[Topic]:
        """Creates a list of Topic objects from JSON file.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).
            use_extended (optional): If true, uses extended topic file version.
              Defaults to True.

        Returns:
            Extracted Topics.
        """
        with open(
            Topic.get_filepath(year, query_rewrite, use_extended),
            "r",
            encoding="utf8",
        ) as f_in:
            raw_topics = json.load(f_in)

        topics = []
        for raw_topic in raw_topics:
            topic_id = raw_topic.get("number")  # int
            description = raw_topic.get("description")  # str
            title = raw_topic.get("title")  # str

            # Canonical result ID is stored under different fields in the 2020
            # topic files depending on the query rewrite.
            canonical_result_id_field = "canonical_result_id"
            if year == "2020":
                if query_rewrite == QueryRewrite.MANUAL:
                    canonical_result_id_field = "manual_canonical_result_id"
                elif query_rewrite == QueryRewrite.AUTOMATIC:
                    canonical_result_id_field = "automatic_canonical_result_id"

            turns = [
                Turn(
                    turn_id=raw_turn.get("number"),
                    canonical_result_id=raw_turn.get(canonical_result_id_field),
                    result_turn_dependence=raw_turn.get(
                        "result_turn_dependence"
                    ),
                    query_turn_dependence=raw_turn.get("query_turn_dependence"),
                    raw_utterance=raw_turn.get("utterance")
                    if raw_turn.get("raw_utterance") is None
                    else raw_turn.get("raw_utterance"),
                    automatic_rewritten_utterance=raw_turn.get(
                        "automatic_rewritten_utterance"
                    ),
                    manual_rewritten_utterance=raw_turn.get(
                        "manual_rewritten_utterance"
                    ),
                    mi_expanded_utterance=raw_turn.get("mi_expanded_utterance")
                    if raw_turn.get("mi_expanded_utterance")
                    else None,
                    passage_id=raw_turn.get("passage_id"),
                    response=raw_turn.get("response"),
                    # 2022 topics file sometimes specifies only doc_id, with
                    # no passage_id indicatior. In such cases, we are taking
                    # the first passage of the document as a provenance.
                    provenance=[
                        provenance + "-1"
                        if "-" not in provenance and len(provenance) > 0
                        else provenance
                        for provenance in raw_turn.get("provenance")
                    ]
                    if "provenance" in raw_turn
                    else raw_turn.get("provenance"),
                    passage=raw_turn.get("passage"),
                    canonical_passage=raw_turn.get("canonical_passage"),
                    provenance_passages=raw_turn.get("provenance_passages"),
                    turn_leaf_id=raw_turn.get("turn_leaf_id"),
                )
                for raw_turn in raw_topic.get("turn")
            ]
            topics.append(Topic(topic_id, description, title, turns))
        return topics

    @staticmethod
    def load_queries_from_file(
        year: str, query_rewrite: QueryRewrite = None
    ) -> List[Query]:
        """Creates a list of Query objects from topic JSON file.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).

        Returns:
            List of Query objects.
        """
        return [
            query
            for topic in Topic.load_topics_from_file(year, query_rewrite)
            for query in topic.get_queries(query_rewrite)
        ]

    @staticmethod
    def load_queries_with_answers_from_file(
        year: str,
        query_rewrite: QueryRewrite = None,
        use_answer_rewrite: bool = False,
    ) -> Union[
        List[Tuple(Query, Document)], List[Tuple(Query, List[Document])]
    ]:
        """Creates a list of Query objects from topic JSON file.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to load
              (auto/manual). Defaults to None (i.e., raw).
            use_answer_rewrite (optional): If true, the document content is the
              rewritten passage, otherwise its full passage. Defaults to False.

        Returns:
            List of Query objects.
        """
        return [
            (
                topic.get_query(turn.turn_id, query_rewrite),
                Document(
                    turn.canonical_result_id,
                    turn.get_passage_content(use_answer_rewrite),
                )
                if year != "2022"
                else [
                    Document(
                        provenance_id,
                        provenance_content,
                    )
                    for provenance_id, provenance_content in zip(
                        turn.provenance,
                        turn.get_provenance_content(use_answer_rewrite),
                    )
                ],
            )
            for topic in Topic.load_topics_from_file(year, query_rewrite)
            for turn in topic.turns
        ]

    @staticmethod
    def load_contexts_from_file(
        year: str,
        query_rewrite: QueryRewrite = None,
        use_answer_rewrite: bool = False,
    ) -> List[Context]:
        """Loads a list of Context objects for topics from a given year.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to include in
              context (auto/manual). Defaults to None (i.e., raw).
            use_answer_rewrite (optional): If true, the document content is the
              rewritten passage, otherwise its full passage. Defaults to False.

        Returns:
            List of Context objects for each question in each topic in a given
            year.
        """
        return [
            context
            for topic in Topic.load_topics_from_file(year, query_rewrite)
            for context in topic.get_contexts(
                year, query_rewrite, use_answer_rewrite
            )
        ]


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="topic.py")
    # General config
    parser.add_argument(
        "--hostname",
        help="Elasticsearch hostname.",
    )
    return parser.parse_args()


def extend_with_canonical_passages(
    host_name: str,
    index_name: str,
    year: str,
    query_rewrite: QueryRewrite = None,
) -> None:
    """Extends turns with canonical passages.

    Stores extended topics to the json file of the same name with "_extended"
    added to the end.

    Args:
        host_name: Elasticsearch hostname.
        index_name: Elasticsearch index to use for passage retrieval.
        year: Year for which to extend the topic file.
        query_rewrite (optional): Type of query rewrite. Defaults to None.
    """
    passage_loader = PassageLoader(host_name, index_name)
    filepath = Topic.get_filepath(year, query_rewrite, use_extended=False)

    # load original topics
    with open(filepath, "r", encoding="utf8") as f_in:
        raw_topics = json.load(f_in)

    # extend turns with canonical answers
    for i, topic in enumerate(
        Topic.load_topics_from_file(year, query_rewrite, use_extended=False)
    ):
        for turn, raw_turn in zip(topic.turns, raw_topics[i]["turn"]):
            if year == "2022":
                if turn.provenance is not None:
                    raw_turn["provenance_passages"] = [
                        passage_loader.get(provenance_id)
                        for provenance_id in turn.provenance
                    ]
            else:
                raw_turn["canonical_passage"] = passage_loader.get(
                    turn.canonical_result_id
                )

    # save extended topics
    with open(
        f"{filepath.split('.json')[0]}_extended.json", "w", encoding="utf8"
    ) as f_out:
        json.dump(raw_topics, f_out)


if __name__ == "__main__":
    args = parse_args()
    opts = {
        "2020": "ms_marco_trec_car_clean",
        "2021": "ms_marco_kilt_wapo_clean",
        "2022": "ms_marco_v2_kilt_wapo_new",
    }
    for year, index_name in opts.items():
        for query_rewrite in [
            QueryRewrite.AUTOMATIC,
            QueryRewrite.MANUAL,
            QueryRewrite.MIXED_INITIATIVE,
        ]:
            extend_with_canonical_passages(
                args.hostname, index_name, year, query_rewrite
            )
