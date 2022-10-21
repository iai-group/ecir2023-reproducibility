"""Tests Loading classes from load_content.py.

These tests expect Elasticsearch on localhost:9204, as well as default files in
locations specified by keyword argument default values or otherwise hardcoded.
"""

from treccast.core.util.passage_loader import PassageLoader


def test_passage_loading():
    ploader = PassageLoader()
    temp_doc_id = "CAR_3add84966af079ed84e8b2fc412ad1dc27800127"
    temp_passage = (
        "Mechanical garage door openers can pull or push a garage door with"
        " enough force to injure or kill people and pets if they become"
        " trapped. All modern openers are equipped with “force settings” that"
        " make the door reverse if it encounters too much resistance while"
        " closing or opening. Any garage door opener sold in the United States"
        " after 1992 requires safety eyes—sensors that prevent the door from"
        " closing if obstructed. Force settings should cause a door to stop or"
        " reverse on encountering more than approximately 20 lbs (9.07 kg) of"
        " resistance. Safety eyes should be installed a maximum of six inches"
        " above the ground. Many garage door injuries, and nearly all garage"
        " door related property damage, can be avoided by following these"
        " precautions."
    )
    assert ploader.get(temp_doc_id) == temp_passage
