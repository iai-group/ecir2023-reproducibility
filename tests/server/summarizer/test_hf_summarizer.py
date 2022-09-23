"""Tests HuggingFaceSummarizer class from hf_summarizer.py"""

import pytest
from treccast.core.ranking import Ranking
from treccast.summarizer.hf_summarizer import HuggingFaceSummarizer


@pytest.fixture
def summarizer() -> HuggingFaceSummarizer:
    return HuggingFaceSummarizer()


@pytest.mark.parametrize(
    "k, min_length, max_length",
    [
        (1, 10, 50),
        (2, 25, 250),
    ],
)
def test_summarize_passages(
    summarizer: HuggingFaceSummarizer,
    ranking: Ranking,
    k: int,
    min_length: int,
    max_length: int,
) -> None:
    summary = summarizer.summarize_passages(ranking, k, min_length, max_length)
    len_summary = len(summarizer._summarizer.tokenizer.tokenize(summary))

    assert len_summary >= min_length
    assert len_summary <= max_length
