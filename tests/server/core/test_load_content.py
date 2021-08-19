"""Tests Loading classes from load_content.py.

These tests expect Elasticsearch on localhost:9204, as well as default files in
locations specified by keyword argument default values or otherwise hardcoded.
"""

from treccast.core.query import Query
from treccast.core.util.load_content import (
    PassageLoader,
    QueryLoader,
    QrelsLoader,
    RunfileLoader,
)


def test_passage_loading():
    ploader = PassageLoader()
    temp_doc_id = "CAR_3add84966af079ed84e8b2fc412ad1dc27800127"
    temp_passage = "Mechanical garage door openers can pull or push a garage door with enough force to injure or kill people and pets if they become trapped. All modern openers are equipped with “force settings” that make the door reverse if it encounters too much resistance while closing or opening. Any garage door opener sold in the United States after 1992 requires safety eyes—sensors that prevent the door from closing if obstructed. Force settings should cause a door to stop or reverse on encountering more than approximately 20 lbs (9.07 kg) of resistance. Safety eyes should be installed a maximum of six inches above the ground. Many garage door injuries, and nearly all garage door related property damage, can be avoided by following these precautions."
    assert ploader.get(temp_doc_id) == temp_passage


def test_query_loading():
    qloader = QueryLoader()
    temp_query = qloader.get("82_1")
    assert type(temp_query) == Query
    assert (
        temp_query.question == "I would like to learn about GMO Food labeling."
    )


def test_ranking_loading():
    qrloader = QrelsLoader()
    query_id = "82_1"
    ranking = qrloader.get_ranking(query_id)
    doc_ids, _ = ranking.documents()
    assert "MARCO_888484" in doc_ids


def test_runfile_loader():
    qploader = RunfileLoader()
    query = qploader.get_query("82_1")
    ranking = qploader.get_ranking("82_1")
    doc_ids, contents = ranking.documents()
    passage = "India and China are the two largest producers of genetically modified products in Asia. India currently only grows GM cotton, while China produces GM varieties of cotton, poplar, petunia, tomato, papaya and sweet pepper. study investigating voluntary labeling in South Africa found that 31% of products labeled as GMO-free had a GM content above 1.0%. In Canada and the USA labeling of GM food is voluntary, while in Europe all food (including processed food) or feed which contains greater than 0.9% of approved GMOs must be labelled."
    assert query.question == "I would like to learn about GMO Food labeling."
    assert "MARCO_888484" in doc_ids
    assert passage in contents
