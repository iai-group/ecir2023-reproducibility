"""Topic class represents conversation as sequence of Turns each consisting
of a Question and Context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import List

from treccast.core.base import Context, Document, Query


class QueryRewrite(Enum):
    AUTOMATIC = 1
    MANUAL = 2


@dataclass
class Turn:
    turn_id: int
    canonical_result_id: str
    result_turn_dependence: int
    query_turn_dependence: list
    raw_utterance: str
    automatic_rewritten_utterance: str = None
    manual_rewritten_utterance: str = None
    passage_id: str = None

    def __post_init__(self):
        if self.passage_id:
            self.canonical_result_id = (
                f"{self.canonical_result_id}-{self.passage_id}"
            )

    # TODO: Extend with the text of the canonical result
    # See https://github.com/iai-group/trec-cast-2021/issues/17

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

        if not utterance:
            raise ValueError(
                "Requested query rewrite is unavailable for this turn "
                f"(#{self.turn_id})"
            )

        return utterance


@dataclass
class Topic:
    topic_id: int
    description: str
    title: str
    turns: List[Turn]

    def get_turn(self, turn_id: int) -> Turn:
        """Returns a given topic turn.

        Args:
            turn_id: Turn ID.

        Raises:
            IndexError: When turn ID is out of bounds.

        Returns:
            Turn instance.
        """
        if len(self.turns) < turn_id:
            raise IndexError(f"Invalid turn_id: {turn_id}")
        return self.turns[turn_id - 1]

    def get_query_id(self, turn_id: int) -> str:
        """Returns query ID corresponding to a given topic turn.

        Args:
            turn_id: Turn ID.

        Returns:
            Query ID.
        """
        return f"{self.topic_id}_{turn_id}"

    def get_query(
        self, turn_id: int, query_rewrite: QueryRewrite = None
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
        return Query(self.get_query_id(turn_id), utterance)

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

    def get_contexts(self, query_rewrite: QueryRewrite = None) -> List[Context]:
        """Gets a list of contexts for each turn.

        Args:
            query_rewrite (optional): Query rewrite variant to include in context
              (auto/manual). Defaults to None (i.e., raw).

        Returns:
            List of Contexts corresponding to every turn with canonical
            responses included.
        """
        queries = self.get_queries(query_rewrite)[:-1]
        canonical_response_ids = [
            turn.canonical_result_id for turn in self.turns
        ][:-1]
        contexts = [None]
        for (query, canonical_response) in zip(queries, canonical_response_ids):
            context = Context()
            context.history = (
                contexts[-1].history.copy() if len(contexts) > 1 else []
            )
            context.history.append((query, Document(canonical_response)))
            contexts.append(context)
        return contexts

    @staticmethod
    def get_filepath(year: str, query_rewrite: QueryRewrite = None) -> str:
        """Returns file path to topic file to be used based on year and query
        rewriting applied.

        Args:
            year: Year (2019_train, 2019, 2020, or 2021).
            query_rewrite (optional): Query rewrite variant to load
                (auto/manual). Defaults to None (i.e., raw).

        Returns:
            Topic file path (relative to repo root).
        """
        filepath = f"data/topics/{year}/"
        if year == "2019_train":
            filepath += "train_topics"
        else:
            variant = (
                "automatic"
                if query_rewrite == QueryRewrite.AUTOMATIC
                else "manual"
            )
            filepath += f"{year}_{variant}_evaluation_topics"
        filepath += "_v1.0.json"

        return filepath

    @staticmethod
    def load_topics_from_file(
        year: str, query_rewrite: QueryRewrite = None
    ) -> List[Topic]:
        """Creates a list of Topic objects from JSON file.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to load
                (auto/manual). Defaults to None (i.e., raw).

        Returns:
            Extracted Topics.
        """
        with open(
            Topic.get_filepath(year, query_rewrite), "r", encoding="utf8"
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
                    raw_utterance=raw_turn.get("raw_utterance"),
                    automatic_rewritten_utterance=raw_turn.get(
                        "automatic_rewritten_utterance"
                    ),
                    manual_rewritten_utterance=raw_turn.get(
                        "manual_rewritten_utterance"
                    ),
                    passage_id=raw_turn.get("passage_id"),
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
    def load_contexts_from_file(
        year: str, query_rewrite: QueryRewrite = None
    ) -> List[Context]:
        """Loads a list of Context objects for topics from a given year.

        Args:
            year: Year.
            query_rewrite (optional): Query rewrite variant to include in context
              (auto/manual). Defaults to None (i.e., raw).

        Returns:
            List of Context objects for each question in each topic in a given
            year.
        """
        return [
            context
            for topic in Topic.load_topics_from_file(year, query_rewrite)
            for context in topic.get_contexts(query_rewrite)
        ]
