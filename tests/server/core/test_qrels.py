"""Tests Qrels class. Requires Elasticsearch instance to use PassageLoader."""

from treccast.core.util.load_content import PassageLoader
from treccast.core.qrel import Qrel


def test_load_qrels():
    ploader = PassageLoader()
    qrels = Qrel.load_qrels_from_file("data/qrels/2020.txt", ploader)
    qrel = qrels["82_1"]
    docs = qrel.documents()
    assert len(docs.keys()) == 1172 - 1044


def test_relevance_levels():
    ploader = PassageLoader()
    qrels = Qrel.load_qrels_from_file("data/qrels/2020.txt", ploader)
    qrel = qrels["82_1"]
    docs = qrel.documents()
    relevant = qrel.get_docs(1)
    assert len(relevant) < len(docs)
    assert (
        relevant[0]["content"]
        == "One of the key issues concerning regulators is whether GM products should be labeled. Labeling can be mandatory up to a threshold GM content level (which varies between countries) or voluntary. A study investigating voluntary labeling in South Africa found that 31% of products labeled as GMO-free had a GM content above 1.0%. In Canada and the United States labeling of GM food is voluntary, while in Europe all food (including processed food) or feed which contains greater than 0.9% of approved GMOs must be labelled. Japan, Malaysia, New Zealand, and Australia require labeling so consumers can exercise choice between foods that have genetically modified, conventional or organic origins."
    )
