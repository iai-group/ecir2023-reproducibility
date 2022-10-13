"""Abstract query rewriting interface."""
import csv
from abc import ABC, abstractmethod

from treccast.core.base import Context, Query


class Rewriter(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def rewrite_query(self, query: Query, context: Context) -> Query:
        """Rewrites a query based on context.

        Args:
            query: Unanswered query (last user utterance) to be
                rewritten.
            context: Conversation context up to the unanswered query.

        Returns:
            Rewritten query.
        """
        raise NotImplementedError


class CachedRewriter(Rewriter):
    def __init__(self, filepath: str) -> None:
        """Simple "rewriter". Loads rewrites from a file.

        Args:
            filepath: Filepath containing rewrites.
        """
        self._get_rewrites(filepath)

    def rewrite_query(self, query: Query, context: Context = None) -> Query:
        """Returns a new query containing a rewrite.

        Args:
            query: Query containing the original question.
            context (optional): Query context. Not used in this class Defaults
                to None.

        Returns:
            A new query with the same query_id containing a rewrite.
        """
        return self._rewrites.get(query.query_id)

    def _get_rewrites(self, filepath: str) -> None:
        """Loads rewrites from a file and stores them into a dictionary.

        The rewrite file should be a TSV file with the following fields:
        Topic ID, Turn ID, Query ID, Rewrite, and Original query.

        Args:
            filepath: Path to the file containing rewrites.
        """
        self._rewrites = {}
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                query = Query(row["id"], row["query"])

                self._rewrites[row["id"]] = query
