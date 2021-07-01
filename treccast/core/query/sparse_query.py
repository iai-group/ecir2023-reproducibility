"""Represents a sparse query, which is a sequence of terms."""

from typing import List

from nltk.corpus import stopwords

from treccast.core.query.query import Query


PUNCTUATION = "!#$%&()*+,./:;<=>?{|}~"
STOPWORDS = stopwords.words("english")


class SparseQuery(Query):
    def __init__(self, question: str) -> None:
        """Initializes a sparse query from a question string.

        Args:
            question (str): Question (raw user utterance).
        """
        # TODO: Consider using an ElasticSearch Analyzer instead.
        # See https://github.com/iai-group/trec-cast-2021/issues/11
        self.__terms = self.__preprocess_question(question)

    def __preprocess_question(self, question: str) -> List[str]:
        """Preprocesses a question by removing punctuation, lowercasing, and
        stopwords removal.

        Args:
            question (str): Question (user utterance).

        Returns:
            str: Sequence of keywords.
        """
        # Remove punctunation.
        question = question.translate(str.maketrans("", "", PUNCTUATION))
        # Lowercasing and splitting.
        return [
            term for term in question.lower().split() if term not in STOPWORDS
        ]

    @property
    def terms(self) -> List[str]:
        return self.__terms
