"""Abstract representation of a query."""

from abc import ABC


class Query(ABC):
    def __init__(self, id: str, text: str) -> None:
        self.__id = id
        self.__text = text

    @property
    def id(self) -> str:
        return self.__id

    @property
    def text(self) -> str:
        return self.__text
