"""Tests for Topic."""

from treccast.core.topic import QueryRewrite, Topic

import pytest


def test_get_filepath():
    assert (
        Topic.get_filepath("2019")
        == "data/topics/2019/2019_manual_evaluation_topics_v1.0.json"
    )
    assert (
        Topic.get_filepath("2020", QueryRewrite.MANUAL)
        == "data/topics/2020/2020_manual_evaluation_topics_v1.0.json"
    )
    assert (
        Topic.get_filepath("2021", QueryRewrite.AUTOMATIC)
        == "data/topics/2021/2021_automatic_evaluation_topics_v1.0.json"
    )


@pytest.fixture
def topics_2020_raw():
    return Topic.load_topics_from_file("2020")


@pytest.fixture
def topics_2020_manual():
    return Topic.load_topics_from_file("2020", QueryRewrite.MANUAL)


@pytest.fixture
def topics_2020_automatic():
    return Topic.load_topics_from_file("2020", QueryRewrite.AUTOMATIC)


@pytest.fixture
def queries_2020_manual():
    return Topic.load_queries_from_file("2020", QueryRewrite.MANUAL)


@pytest.fixture
def queries_2020_automatic():
    return Topic.load_queries_from_file("2020", QueryRewrite.AUTOMATIC)


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


def test_get_query_raw(topics_2020_raw):
    query = topics_2020_raw[0].get_query(2)
    assert query.query_id == "81_2"
    assert query.question == "Now it stopped working. Why?"


def test_get_query_manual_rewrite_via_topics(topics_2020_manual):
    query = topics_2020_manual[0].get_query(2, QueryRewrite.MANUAL)
    assert query.query_id == "81_2"
    assert query.question == "Now my garage door opener stopped working. Why?"


def test_get_query_manual_rewrite_via_queries(queries_2020_manual):
    query = queries_2020_manual[1]
    assert query.query_id == "81_2"
    assert query.question == "Now my garage door opener stopped working. Why?"


def test_get_query_automatic_rewrite_via_topics(topics_2020_automatic):
    query = topics_2020_automatic[0].get_query(2, QueryRewrite.AUTOMATIC)
    assert query.query_id == "81_2"
    assert query.question == "Why did garage door opener stop working?"


def test_get_query_automatic_rewrite_via_queries(queries_2020_automatic):
    query = queries_2020_automatic[1]
    assert query.query_id == "81_2"
    assert query.question == "Why did garage door opener stop working?"
