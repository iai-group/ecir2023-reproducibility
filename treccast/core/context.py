"""Represents conversation context.

It contains the previous dialogue turns as well as the respective system
responses (if available).
"""

from typing import NamedTuple

Turn = NamedTuple("Turn", [("user_utterance", str), ("system_utterance", str)])


class Context:
    def __init__(self) -> None:
        self.__turns = []