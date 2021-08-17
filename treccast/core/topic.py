"""Topic class represents conversation as sequence of Turns each consisting
of a Question and Context.
"""

import json

from typing import List, Tuple
from dataclasses import dataclass


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

    def get_question_and_context(
        self, turn_id: int, utterance_type: str = "raw"
    ) -> Tuple[str, List[Turn]]:
        """Returns the question and context for a given topic turn.

        Args:
            turn_id: Turn ID.
            utterance_type: Type of utterance to return. Choices are "raw",
                "manual", and "automatic".

        Returns:
            Question (str) and list of Turns.
        """
        turn = self.get_turn(turn_id)
        if utterance_type == "raw":
            utterance = turn.raw_utterance
        elif utterance_type == "automatic":
            utterance = turn.automatic_rewritten_utterance
        elif utterance_type == "manual":
            utterance = turn.manual_rewritten_utterance
        else:
            raise ValueError(
                "Incorrect utterance type. Accepted values are: 'raw', 'manual'"
                ", and 'automatic'."
            )
        return utterance, self.turns[: turn_id - 1]


def construct_topics_from_file(filepath: str) -> List[Topic]:
    """Creates a list of Topic objects from JSON file.

    Args:
        filepath: Path to JSON file with topics.

    Returns:
        Extracted Topics.
    """
    with open(filepath, "r", encoding="utf8") as f_in:
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
                result_turn_dependence=raw_turn.get("result_turn_dependence"),
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
