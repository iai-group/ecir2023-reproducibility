import pytest
from treccast.indexer.indexer import Indexer

MS_MARCO_PASSAGE_DATASET = "tests/data/ms_marco_passage_sample.tsv"
TREC_CAR_PARAGRAPH_DATASET = "tests/data/trec_car_paragraph_sample.cbor"
INDEX_NAME = "test_ms_marco_trec_car"


@pytest.fixture
def indexer():
    return Indexer(INDEX_NAME)


def test_generate_data_marco_first_entry(indexer):
    generator = indexer.generate_data_marco(MS_MARCO_PASSAGE_DATASET)
    result = next(generator)

    assert result["_id"] == "MARCO_0"
    assert result["_index"] == INDEX_NAME
    assert result["_source"] == {
        "body": "The presence of communication amid "
        "scientific minds was equally important to the success of the Manhattan"
        " Project as scientific intellect was. The only cloud hanging over the "
        "impressive achievement of the atomic researchers and engineers is what"
        " their success truly meant; hundreds of thousands of innocent lives "
        "obliterated."
    }


def test_generate_data_marco_last_entry(indexer):
    generator = indexer.generate_data_marco(MS_MARCO_PASSAGE_DATASET)
    result = list(generator)[-1]

    assert result["_id"] == "MARCO_99"
    assert result["_index"] == INDEX_NAME
    assert result["_source"] == {
        "body": "(1841 - 1904) Contrary to legend, AntonÃ­n DvoÅÃ¡k (September"
        " 8, 1841 - May 1, 1904) was not born in poverty. His father was an "
        "innkeeper and butcher, as well as an amateur musician. The father not "
        "only put no obstacles in the way of his son's pursuit of a musical "
        "career, he and his wife positively encouraged the boy."
    }


def test_generate_data_car(indexer):
    generator = indexer.generate_data_car(TREC_CAR_PARAGRAPH_DATASET)
    result = next(generator)

    assert result["_id"] == "CAR_00000047dc43083f49b68399c6deeed5c0e81c1f"
    assert result["_index"] == INDEX_NAME
    assert result["_source"] == {
        "body": "On 28 October 1943, Fuller sailed from Efate, New Hebrides, "
        "for the initial landings on Bougainville, where she landed Marine "
        "raiders on Cape Torokina 1 November. Laden with casualties, she "
        "cleared the assault beaches the same day for Tulagi and Purvis Bay. "
        "Returning to Bougainville's Empress Augusta Bay anchorage with "
        "reinforcements 8 November, Fuller came under enemy air attack, "
        "receiving a direct hit on her port side which set her afire and killed"
        " five of her crew and two soldiers embarked. She returned to Purvis "
        "Bay 2 days later to repair battle damage, and twice more during the "
        "following month and a half carried reinforcements to Bougainville."
    }


def test_generate_data_car_last_entry(indexer):
    generator = indexer.generate_data_car(TREC_CAR_PARAGRAPH_DATASET)
    result = list(generator)[-1]

    assert result["_id"] == "CAR_0000404a1797b531c6f7da0dde5743b78305cf88"
    assert result["_index"] == INDEX_NAME
    assert result["_source"] == {
        "body": "Mannert was born in Altdorf bei Nürnberg, where he did his "
        "studies. In 1784 he became a teacher at the Sebaldusschule in "
        "Nuremberg, and in 1788 at the Ägidiusgymnasium there. In 1796 he "
        "became professor of history at the University of Altdorf, in 1805 at "
        "the University of Würzburg, in 1807 at the Ludwig Maximilian University"
        " of Munich (then in Landshut), and from 1826 at the same university in"
        " its new location in Munich. He died in Munich in 1834. His historical"
        " work was known in particular for its focus on studying primary "
        "sources.",
    }