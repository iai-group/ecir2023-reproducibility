"""Tests for Topic."""

from typing import List

import pytest
from treccast.core.topic import Context, Document, QueryRewrite, Topic


@pytest.mark.parametrize(
    ("year", "query_rewrite", "use_extended", "file_path"),
    [
        (
            "2019",
            QueryRewrite.MANUAL,
            False,
            "data/topics/2019/2019_manual_evaluation_topics_v1.0.json",
        ),
        (
            "2020",
            QueryRewrite.MANUAL,
            True,
            "data/topics/2020/2020_manual_evaluation_topics_v1.0_extended.json",
        ),
        (
            "2021",
            QueryRewrite.AUTOMATIC,
            True,
            "data/topics/2021/2021_automatic_evaluation_topics_v1.0_extended"
            ".json",
        ),
        (
            "2022",
            QueryRewrite.AUTOMATIC,
            True,
            "data/topics/2022/2022_automatic_evaluation_topics_v1.0_extended"
            ".json",
        ),
    ],
)
def test_get_filepath(year, query_rewrite, use_extended, file_path):
    assert Topic.get_filepath(year, query_rewrite, use_extended) == file_path


@pytest.fixture
def topics_2020_manual() -> List[Topic]:
    return Topic.load_topics_from_file("2020", QueryRewrite.MANUAL)


def test_load_topics(topics_2020_manual):
    assert len(topics_2020_manual) == 25


def test_topic_construction(topics_2020_manual):
    topic = topics_2020_manual[0]
    assert topic.topic_id == 81
    assert type(topic.turns) == list
    assert len(topic.turns) == 8


def test_get_first_turn(topics_2020_manual):
    turn = topics_2020_manual[0].get_turn(1)
    assert turn.turn_id == 1
    assert (
        turn.raw_utterance
        == "How do you know when your garage door opener is going bad?"
    )


def test_get_last_turn(topics_2020_manual):
    turn = topics_2020_manual[1].get_turn(10)
    assert turn.turn_id == 10
    assert turn.raw_utterance == "How could Co-Extra improve it?"
    assert (
        turn.manual_rewritten_utterance
        == "How could Co-Extra improve DNA-based testing for GMO contamination?"
    )


def test_get_invalid_turn(topics_2020_manual):
    with pytest.raises(IndexError):
        topics_2020_manual[1].get_turn(11)


def test_get_query_raw():
    topic = Topic.load_topics_from_file("2020")[0]
    query = topic.get_query(2)
    assert query.query_id == "81_2"
    assert query.question == "Now it stopped working. Why?"


def test_get_query_manual_rewrite_via_topics(topics_2020_manual):
    topic = topics_2020_manual[0]
    query = topic.get_query(2, QueryRewrite.MANUAL)
    assert query.query_id == "81_2"
    assert query.question == "Now my garage door opener stopped working. Why?"


def test_get_query_manual_rewrite_via_queries():
    query = Topic.load_queries_from_file("2020", QueryRewrite.MANUAL)[1]
    assert query.query_id == "81_2"
    assert query.question == "Now my garage door opener stopped working. Why?"


def test_get_query_automatic_rewrite_via_topics():
    topic = Topic.load_topics_from_file("2020", QueryRewrite.AUTOMATIC)[0]
    query = topic.get_query(2, QueryRewrite.AUTOMATIC)
    assert query.query_id == "81_2"
    assert query.question == "Why did garage door opener stop working?"


def test_get_query_automatic_rewrite_via_queries():
    query = Topic.load_queries_from_file("2020", QueryRewrite.AUTOMATIC)[1]
    assert query.query_id == "81_2"
    assert query.question == "Why did garage door opener stop working?"


@pytest.mark.parametrize(
    "query_rewrite", [None, QueryRewrite.MANUAL, QueryRewrite.AUTOMATIC]
)
@pytest.mark.parametrize("year", ["2020", "2021"])
def test_get_context(year, query_rewrite):
    topic = Topic.load_topics_from_file(year, query_rewrite)[0]
    query_1 = topic.get_query(1, query_rewrite)
    canonical_response = topic.turns[0].canonical_passage
    query_2 = topic.get_query(2, query_rewrite)
    canonical_response_2 = topic.turns[1].canonical_passage
    contexts = topic.get_contexts(year, query_rewrite)
    assert contexts[0] is None
    context_2 = Context()
    context_2.history = [(query_1, [Document(None, canonical_response)])]
    assert contexts[1] == context_2
    context_3 = Context()
    context_3.history = context_2.history
    context_3.history.append((query_2, [Document(None, canonical_response_2)]))
    assert contexts[2] == context_3


@pytest.mark.parametrize(
    "query_rewrite", [None, QueryRewrite.MANUAL, QueryRewrite.AUTOMATIC]
)
def test_get_context_2022(query_rewrite):
    topic = Topic.load_topics_from_file("2022", query_rewrite)[0]
    query_1 = topic.get_query("1-1", query_rewrite)
    canonical_responses = topic.turns[0].provenance_passages
    query_2 = topic.get_query("1-3", query_rewrite)
    canonical_responses_2 = topic.turns[1].provenance_passages
    contexts = topic.get_contexts("2022", query_rewrite)
    assert contexts[0] is None
    context_2 = Context()
    context_2.history = [
        (
            query_1,
            [Document(None, response) for response in canonical_responses],
        )
    ]
    assert contexts[1] == context_2
    context_3 = Context()
    context_3.history = context_2.history
    context_3.history.append(
        (
            query_2,
            [Document(None, response) for response in canonical_responses_2],
        )
    )
    assert contexts[2] == context_3
