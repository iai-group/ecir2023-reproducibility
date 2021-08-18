"""Topic class represents conversation as sequence of Turns each consisting
of a Question and Context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import List

from treccast.core.query.query import Query


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
        query_id = f"{self.topic_id}_{turn_id}"
        utterance = self.get_turn(turn_id).get_utterance(query_rewrite)
        return Query(query_id, utterance)

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
        elif year == "2019":
            filepath += "evaluation_topics"
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
            turns = [
                Turn(
                    turn_id=raw_turn.get("number"),
                    canonical_result_id=raw_turn.get("canonical_result_id"),
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
