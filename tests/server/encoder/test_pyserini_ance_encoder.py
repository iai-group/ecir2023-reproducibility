"""Tests PyseriniAnceEncoder class from pyserini_ance_encoder.py"""

from contextlib import nullcontext as does_not_raise
import os

import pytest
from treccast.encoder.pyserini_ance_encoder import PyseriniAnceEncoder
from typing import Any

_BATCH_SIZE = 80
_MS_MARCO_PASSAGE_DATASET = "tests/data/ms_marco_passage_sample.tar.gz"
_TREC_CAR_PARAGRAPH_DATASET = "tests/data/trec_car_paragraph_sample.cbor"
_MS_MARCO_PASSAGE_DATASET_TRECWEB = "tests/data/ms_marco_sample.trecweb"

_ACTION = "encoding"


@pytest.fixture
def encoder() -> PyseriniAnceEncoder:
    enc = PyseriniAnceEncoder(batch_size=_BATCH_SIZE)
    assert enc._batch_size == _BATCH_SIZE
    assert enc._embedding_dim == 768
    return enc


def test_generate_data_marco(encoder: PyseriniAnceEncoder) -> None:
    generator = encoder.generate_data_marco(_ACTION, _MS_MARCO_PASSAGE_DATASET)
    result = next(generator)

    assert result[0] == "MARCO_0"
    assert (
        result[1] == "The presence of communication amid scientific minds was "
        "equally important to the success of the Manhattan Project as "
        "scientific intellect was. The only cloud hanging over the impressive "
        "achievement of the atomic researchers and engineers is what their "
        "success truly meant; hundreds of thousands of innocent lives "
        "obliterated."
    )


def test_generate_data_car(encoder: PyseriniAnceEncoder) -> None:
    generator = encoder.generate_data_car(_ACTION, _TREC_CAR_PARAGRAPH_DATASET)
    result = next(generator)

    assert result[0] == "CAR_00000047dc43083f49b68399c6deeed5c0e81c1f"
    assert (
        result[1] == "On 28 October 1943, Fuller sailed from Efate, New "
        "Hebrides, for the initial landings on Bougainville, where she landed "
        "Marine raiders on Cape Torokina 1 November. Laden with casualties, she"
        " cleared the assault beaches the same day for Tulagi and Purvis Bay. "
        "Returning to Bougainville's Empress Augusta Bay anchorage with "
        "reinforcements 8 November, Fuller came under enemy air attack, "
        "receiving a direct hit on her port side which set her afire and killed"
        " five of her crew and two soldiers embarked. She returned to Purvis "
        "Bay 2 days later to repair battle damage, and twice more during the "
        "following month and a half carried reinforcements to Bougainville."
    )


def test_generate_data_trecweb(encoder: PyseriniAnceEncoder) -> None:
    generator = encoder.generate_data_trecweb(
        _ACTION, _MS_MARCO_PASSAGE_DATASET_TRECWEB
    )
    result = list(generator)[-1]

    assert result[0] == "MARCO_D301595-6"
    assert (
        result[1] == "Developmental Milestones and Your <8>-Year-Old Child "
        "A Word From Verywell Your 8-year-old is in the full bloom of childhood"
        ". Enjoy activities and explore the world together. It's a great time "
        "to spark new interests in your child and watch her grow in every way. "
        "Sources: Anthony, Michelle. The emotional lives of 8-10-year-olds. "
        "Scholastic Publishing. Chaplin TM, Aldao A. Gender differences in "
        "emotion expression in children: A meta-analytic review Psychological "
        "Bulletin. 2013;139 (4):735-765. doi:10.1037/a0030737. Middle childhood"
        '. CDC. "'
    )


def test_generate_batches(encoder: PyseriniAnceEncoder) -> None:
    generator = encoder.generate_data_marco(_ACTION, _MS_MARCO_PASSAGE_DATASET)
    batches_generator = encoder.generate_batches(generator, encoder._batch_size)

    batch = next(batches_generator)
    assert len(batch) == 80
    batch = next(batches_generator)
    assert len(batch) == 20


def test_encode(encoder: PyseriniAnceEncoder) -> None:
    text = "The presence of communication amid scientific minds was equally "
    "important to the success of the Manhattan Project as scientific intellect "
    "was. The only cloud hanging over the impressive achievement of the atomic "
    "researchers and engineers is what their success truly meant; hundreds of "
    "thousands of innocent lives obliterated."

    embeddings = encoder.encode(text)
    assert embeddings.shape == (1, 768)


@pytest.mark.parametrize(
    "filepath, expectation",
    [
        ("tests/data/output_embeddings.hdf5", does_not_raise()),
        ("tests/data/output_embeddings.gz", pytest.raises(ValueError)),
    ],
)
def test_save(
    encoder: PyseriniAnceEncoder, filepath: str, expectation: Any
) -> None:
    text = "The presence of communication amid scientific minds was equally "
    "important to the success of the Manhattan Project as scientific intellect "
    "was. The only cloud hanging over the impressive achievement of the atomic "
    "researchers and engineers is what their success truly meant; hundreds of "
    "thousands of innocent lives obliterated."

    embeddings = encoder.encode(text)

    with expectation:
        encoder.save(filepath, ["MARCO_0"], embeddings)
        os.remove(filepath)
