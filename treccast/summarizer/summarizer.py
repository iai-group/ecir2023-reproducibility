"""Abstract summarization interface."""

from abc import ABC, abstractmethod

from treccast.core.ranking import Ranking


class Summarizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def summarize_passages(
        self, passages: Ranking, k: int, min_length: int, max_length: int
    ) -> str:
        """Summarize top-k passages to generate an answer.

        Args:
            passages: Passages to summarize.
            k: Maximum number of passages to consider for the summary.
            min_length: Minimum number of tokens in the summary.
            max_length: Maximum number of tokens in the summary.

        Returns:
            Summary of passages.
        """
        raise NotImplementedError
