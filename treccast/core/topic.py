"""Topic class represents conversation as sequence of Turns each consisting
of a Question and Context.
"""

import json

from typing import List, Tuple


class Turn:
    def __init__(
        self,
        result_turn_dependence: int,
        canonical_result_id: str,
        turn_id: int,
        manual_rewritten_utterance: str,
        automatic_rewritten_utterance: str,
        query_turn_dependence: list,
        raw_utterance: str,
    ) -> None:
        """Create a Turn object with Args as Attributes.

        Args:
            result_turn_dependence: [description]
            canonical_result_id: [description]
            turn_id: [description]
            manual_rewritten_utterance: [description]
            automatic_rewritten_utterance: [description]
            query_turn_dependence: [description]
            raw_utterance: [description]
        """
        self._result_turn_dependence = result_turn_dependence
        self._canonical_result_id = canonical_result_id
        self._turn_id = turn_id
        self._manual_rewritten_utterance = manual_rewritten_utterance
        self._automatic_rewritten_utterance = automatic_rewritten_utterance
        self._query_turn_dependence = query_turn_dependence
        self._raw_utterance = raw_utterance
        # TODO: Extend with the text of the canonical result
        # See https://github.com/iai-group/trec-cast-2021/issues/17

    @property
    def turn_id(self) -> int:
        return self._turn_id

    @property
    def raw_utterance(self) -> str:
        return self._raw_utterance

    @property
    def manual_utterance(self) -> str:
        return self._manual_rewritten_utterance

    @property
    def automatic_utterance(self) -> str:
        return self._automatic_rewritten_utterance


class Topic:
    def __init__(
        self, topic_id: int, description: str, title: str, turns: List[Turn]
    ) -> None:
        """Initializes a topic.

        Args:
            topic_id: Topic ID (as provided by TREC).
            description: Topic description.
            title: Topic title.
            turns: List of conversation turns.
        """
        self._topic_id = topic_id
        self._description = description
        self._title = title
        self._turns = turns

    @property
    def topic_id(self) -> int:
        return self._topic_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def title(self) -> str:
        return self._title

    @property
    def turns(self) -> List[Turn]:
        return self._turns

    def get_turn(self, turn_id: int) -> Turn:
        """Returns a given topic turn.

        Args:
            turn_id: Turn ID.

        Raises:
            IndexError: When turn ID is out of bounds.

        Returns:
            Turn instance.
        """
        if len(self._turns) < turn_id:
            raise IndexError(f"Invalid turn_id: {turn_id}")
        return self._turns[turn_id - 1]

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
            utterance = turn.automatic_utterance
        elif utterance_type == "manual":
            utterance = turn.manual_utterance
        else:
            raise ValueError(
                "Incorrect utterance type. Accepted values are: 'raw', 'manual', and 'automatic'."
            )
        return utterance, self._turns[: turn_id - 1]


def construct_topics_from_file(filepath: str) -> List[Topic]:
    """Creates a list of Topic objects from JSON file.

    Args:
        filepath: Path to JSON file with topics.

    Returns:
        Extracted Topics.
    """
    topics = []
    with open(filepath, "r", encoding="utf8") as f_in:
        raw_topics = json.load(f_in)
        for raw_topic in raw_topics:
            topic_id = raw_topic.get("number")  # int
            description = raw_topic.get("description")  # str
            title = raw_topic.get("title")  # str
            raw_turns = raw_topic.get("turn")  # List(dict)
            turns = []
            for raw_turn in raw_turns:
                turn_id = raw_turn.get("number")
                result_turn_dependence = raw_turn.get("result_turn_dependence")
                canonical_result_id = raw_turn.get("canonical_result_id")
                manual_rewritten_utterance = raw_turn.get(
                    "manual_rewritten_utterance"
                )
                automatic_rewritten_utterance = raw_turn.get(
                    "automatic_rewritten_utterance"
                )
                query_turn_dependence = raw_turn.get("query_turn_dependence")
                raw_utterance = raw_turn.get("raw_utterance")
                turn = Turn(
                    result_turn_dependence,
                    canonical_result_id,
                    turn_id,
                    manual_rewritten_utterance,
                    automatic_rewritten_utterance,
                    query_turn_dependence,
                    raw_utterance,
                )
                turns.append(turn)
            topic = Topic(topic_id, description, title, turns)
            topics.append(topic)
    return topics
