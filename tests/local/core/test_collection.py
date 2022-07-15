"""Tests EmbeddingCollection class from collection.

Embeddings sample was created based on:
http://ann-benchmarks.com/glove-100-angular.hdf5
"""

import pytest
import numpy as np
from treccast.core.collection import EmbeddingCollection

_EMBEDDINGS_FILE = "tests/data/embeddings_sample.hdf5"


@pytest.fixture
def embedding_collection() -> EmbeddingCollection:
    return EmbeddingCollection(_EMBEDDINGS_FILE)


def test_load_embeddings_raise_error(
    embedding_collection: EmbeddingCollection,
    filepath="tests/data/embeddings_sample.gzip",
) -> None:
    with pytest.raises(ValueError):
        embedding_collection.load_embeddings(filepath)


def test_properties(embedding_collection: EmbeddingCollection) -> None:
    true_passage_ids = [b"001", b"002", b"003", b"004", b"005"]
    assert true_passage_ids == embedding_collection.passage_ids[:].tolist()

    embedding_4 = np.array(
        [
            -0.73277998,
            -0.47152999,
            1.3312,
            0.94514,
            0.39193001,
            -0.15516999,
            0.77000999,
            -0.48571,
            -0.25486001,
            -0.51172,
            -0.61295998,
            0.85049999,
            1.22290003,
            -0.73391998,
            0.20598,
            -0.32854,
            0.46325001,
            1.68959999,
            1.09179997,
            -0.074555,
            -0.10862,
            0.020611,
            -0.28641,
            1.08449996,
            -0.54579997,
            0.89077997,
            0.23955999,
            0.66657001,
            -0.19411001,
            1.47640002,
            0.72429001,
            -0.41979,
            1.13080001,
            -0.20669,
            0.74269003,
            -0.32853001,
            -0.49173999,
            -0.30232,
            -1.77509999,
            0.28806001,
            0.33039001,
            -0.54277003,
            0.55764002,
            -0.15443,
            0.89683002,
            -0.67372,
            -0.20815,
            -0.77805001,
            1.00909996,
            0.91040999,
            -0.38358,
            0.13209,
            -0.69006002,
            1.93809998,
            -1.16729999,
            -0.49340001,
            0.65948999,
            -0.85698003,
            -1.22809994,
            0.089305,
            -0.67711002,
            0.11726,
            -0.67690003,
            0.96728998,
            -1.31439996,
            -0.88628,
            1.20050001,
            0.57995999,
            0.88940001,
            -0.57029998,
            -1.85399997,
            0.39546001,
            0.59101999,
            0.61267,
            0.69439,
            -1.27629995,
            0.17319,
            -0.48363999,
            -0.09554,
            -0.19799,
            -0.20332,
            -0.061615,
            -1.41830003,
            0.06963,
            0.019058,
            -0.60355997,
            -0.12622,
            -0.29150999,
            0.59030998,
            0.41391999,
            0.46976,
            -0.70778,
            0.35541999,
            1.01590002,
            0.38979,
            0.57077998,
            0.19437,
            0.89064002,
            -0.80821002,
            0.92133999,
        ]
    )

    np.testing.assert_allclose(embedding_4, embedding_collection.embeddings[3])
