"""Tests for Topic."""

from treccast.core.topic import construct_topics_from_file

import pytest


@pytest.fixture
def topics():
    filepath = (
        "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
    )
    return construct_topics_from_file(filepath)


def test_load_topics(topics):
    assert len(topics) == 25


def test_topic_construction(topics):
    topic = topics[0]

    assert topic.topic_id == 81
    assert (
        topic.description
        == "Information on how to replace and repair garage doors."
    )
    assert topic.title == "garage door repair and replacement"
    assert type(topic.turns) == list
    assert len(topic.turns) == 9


def test_get_first_turn(topics):
    turn = topics[0].get_turn(1)
    assert turn.turn_id == 1
    assert (
        turn.raw_utterance
        == "How do you know when your garage door opener is going bad?"
    )


def test_get_last_turn(topics):
    turn = topics[1].get_turn(10)
    assert turn.turn_id == 10
    assert turn.raw_utterance == "How could Co-Extra improve it?"


def test_get_invalid_turn(topics):
    with pytest.raises(IndexError):
        topics[1].get_turn(11)


def test_get_question_and_context_first_turn(topics):
    question, context = topics[0].get_question_and_context(1)
    assert (
        question == "How do you know when your garage door opener is going bad?"
    )
    assert context == []


def test_get_question_and_context_second_turn(topics):
    question, context = topics[0].get_question_and_context(2)
    assert question == "Now it\u0027s stopped working. Why?"
    assert len(context) == 1
    assert (
        context[0].raw_utterance
        == "How do you know when your garage door opener is going bad?"
    )
