"""Tests for Topic."""

import pytest
from typing import List
from treccast.core.topic import Context, Document, QueryRewrite, Topic, Query


def test_get_filepath():
    assert (
        Topic.get_filepath("2019", QueryRewrite.MANUAL, False)
        == "data/topics/2019/2019_manual_evaluation_topics_v1.0.json"
    )
    assert (
        Topic.get_filepath("2020", QueryRewrite.MANUAL, False)
        == "data/topics/2020/2020_manual_evaluation_topics_v1.0.json"
    )
    assert (
        Topic.get_filepath("2021", QueryRewrite.AUTOMATIC, False)
        == "data/topics/2021/2021_automatic_evaluation_topics_v1.0.json"
    )
    assert (
        Topic.get_filepath("2022", QueryRewrite.AUTOMATIC, False)
        == "data/topics/2022/2022_automatic_evaluation_topics_v1.0.json"
    )


@pytest.fixture
def topics_2020_raw() -> List[Topic]:
    return Topic.load_topics_from_file("2020")


@pytest.fixture
def topics_2020_manual() -> List[Topic]:
    return Topic.load_topics_from_file("2020", QueryRewrite.MANUAL)


@pytest.fixture
def topics_2020_automatic() -> List[Topic]:
    return Topic.load_topics_from_file("2020", QueryRewrite.AUTOMATIC)


@pytest.fixture
def topics_2021_raw() -> List[Topic]:
    return Topic.load_topics_from_file("2021")


@pytest.fixture
def topics_2021_manual() -> List[Topic]:
    return Topic.load_topics_from_file("2021", QueryRewrite.MANUAL)


@pytest.fixture
def topics_2021_automatic() -> List[Topic]:
    return Topic.load_topics_from_file("2021", QueryRewrite.AUTOMATIC)


@pytest.fixture
def topics_2022_raw() -> List[Topic]:
    return Topic.load_topics_from_file("2022", None, False)


@pytest.fixture
def topics_2022_manual() -> List[Topic]:
    return Topic.load_topics_from_file("2022", QueryRewrite.MANUAL, False)


@pytest.fixture
def topics_2022_automatic() -> List[Topic]:
    return Topic.load_topics_from_file("2022", QueryRewrite.AUTOMATIC, False)


@pytest.fixture
def queries_2020_manual() -> List[Query]:
    return Topic.load_queries_from_file("2020", QueryRewrite.MANUAL)


@pytest.fixture
def queries_2020_automatic() -> List[Query]:
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


@pytest.mark.parametrize(
    "topics, year, query_rewrite",
    [
        ("topics_2020_raw", "2020", None),
        ("topics_2020_manual", "2020", QueryRewrite.MANUAL),
        ("topics_2020_automatic", "2020", QueryRewrite.AUTOMATIC),
        ("topics_2021_raw", "2021", None),
        ("topics_2021_manual", "2021", QueryRewrite.MANUAL),
        ("topics_2021_automatic", "2021", QueryRewrite.AUTOMATIC),
    ],
)
def test_get_context_raw(topics, year, query_rewrite, request):
    topic = request.getfixturevalue(topics)[0]
    query_1 = topic.get_query(1, query_rewrite)
    canonical_response_id_1 = topic.turns[0].canonical_result_id
    query_2 = topic.get_query(2, query_rewrite)
    canonical_response_id_2 = topic.turns[1].canonical_result_id
    contexts = topic.get_contexts(year, query_rewrite)
    assert contexts[0] is None
    context_2 = Context()
    context_2.history = [(query_1, [Document(canonical_response_id_1)])]
    assert contexts[1] == context_2
    context_3 = Context()
    context_3.history = context_2.history
    context_3.history.append((query_2, [Document(canonical_response_id_2)]))
    assert contexts[2] == context_3


@pytest.mark.parametrize(
    "topics, query_rewrite",
    [
        ("topics_2022_raw", None),
        ("topics_2022_manual", QueryRewrite.MANUAL),
        ("topics_2022_automatic", QueryRewrite.AUTOMATIC),
    ],
)
def test_get_context_raw_2022(topics, query_rewrite, request):
    topic = request.getfixturevalue(topics)[0]
    query_1 = topic.get_query("1-1", query_rewrite)
    canonical_response_ids_1 = topic.turns[0].provenance
    query_2 = topic.get_query("1-3", query_rewrite)
    canonical_response_ids_2 = topic.turns[1].provenance
    contexts = topic.get_contexts("2022", query_rewrite)
    assert contexts[0] is None
    context_2 = Context()
    context_2.history = [
        (
            query_1,
            [
                Document(response_id_1)
                for response_id_1 in canonical_response_ids_1
            ],
        )
    ]
    assert contexts[1] == context_2
    context_3 = Context()
    context_3.history = context_2.history
    context_3.history.append(
        (
            query_2,
            [
                Document(response_id_2)
                for response_id_2 in canonical_response_ids_2
            ],
        )
    )
    assert contexts[2] == context_3
