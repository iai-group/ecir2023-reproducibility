"""Represents a sparse query as a sequence of terms."""

from typing import List

from nltk.corpus import stopwords
import nltk
from treccast.core.query.query import Query


PUNCTUATION = "!#$%&()*+,./:;<=>?{|}~"
# Stop words need to be downloaded if they are not already.
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")


class SparseQuery(Query):
    def __init__(self, query_id: str, question: str) -> None:
        """Initializes a sparse query from a question string.

        Args:
            query_id: Query ID.
            question: Question (raw user utterance).
        """
        super().__init__(query_id, question)
        # TODO: Consider using an ElasticSearch Analyzer instead.
        # See https://github.com/iai-group/trec-cast-2021/issues/11
        self._terms = self._preprocess_question(question)

    def _preprocess_question(self, question: str) -> List[str]:
        """Preprocesses a question by removing punctuation, lowercasing, and
        stopwords removal.

        Args:
            question: Question (user utterance).

        Returns:
            Sequence of keywords.
        """
        # Remove punctunation.
        question = question.translate(str.maketrans("", "", PUNCTUATION))
        # Lowercasing and splitting.
        return [
            term for term in question.lower().split() if term not in STOPWORDS
        ]

    @property
    def terms(self) -> List[str]:
        return self._terms

    @property
    def query_text(self) -> str:
        # TODO: This should be changed to avoid splitting and re-joining.
        # See https://github.com/iai-group/trec-cast-2021/issues/11
        return " ".join(self._terms)
