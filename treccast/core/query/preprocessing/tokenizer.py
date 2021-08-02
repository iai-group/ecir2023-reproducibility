"""Abstract interface of a tokenizer."""

from abc import ABC, abstractstaticmethod
from typing import List

from nltk.corpus import stopwords
import nltk

PUNCTUATION = "!#$%&()*+,./:;<=>?{|}~"
# Stop words need to be downloaded if they are not already.
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")


class Tokenizer(ABC):
    @abstractstaticmethod
    def get_tokens(question: str) -> List[str]:
        """Returns a list of tokens for the given question.

        Args:
            question: Input question that should be tokenized.

        Returns:
            List of tokens.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def preprocess_query(question: str) -> str:
        """Transforms the questions into a pre-processed query.

        Args:
            question: Input question that should be transformed.


        Returns:
            Pre-processed query.
        """

        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
    @staticmethod
    def get_tokens(question: str) -> List[str]:
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

    @staticmethod
    def preprocess_query(question: str) -> str:
        """Concatenation of tokens after lowercasing, removal of punctuation and
        stopwords.

        Args:
            question: Question (user utterance).

        Returns:
            Processed query.
        """
        return " ".join(SimpleTokenizer.get_tokens(question))
