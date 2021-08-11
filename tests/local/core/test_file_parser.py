"""Tests FileParser class from file_parser"""
import pytest

from treccast.core.util.file_parser import FileParser


MS_MARCO_PASSAGE_DATASET_GZ = "tests/data/ms_marco_passage_sample.tar.gz"
MS_MARCO_PASSAGE_DATASET = "tests/data/ms_marco_passage_sample.tsv"


@pytest.mark.parametrize(
    "filepath", [MS_MARCO_PASSAGE_DATASET_GZ, MS_MARCO_PASSAGE_DATASET]
)
def test_parse_first(filepath):
    first_line = next(FileParser.parse(filepath))
    assert first_line == (
        "0\tThe presence of communication amid "
        "scientific minds was equally important to the success of the Manhattan"
        " Project as scientific intellect was. The only cloud hanging over the "
        "impressive achievement of the atomic researchers and engineers is what"
        " their success truly meant; hundreds of thousands of innocent lives "
        "obliterated."
    )


@pytest.mark.parametrize(
    "filepath", [MS_MARCO_PASSAGE_DATASET_GZ, MS_MARCO_PASSAGE_DATASET]
)
def test_parse_last(filepath):
    generator = FileParser.parse(filepath)
    last_line = list(generator)[-1]
    assert last_line == (
        "99\t(1841 - 1904) Contrary to legend, AntonÃ­n DvoÅÃ¡k (September"
        " 8, 1841 - May 1, 1904) was not born in poverty. His father was an "
        "innkeeper and butcher, as well as an amateur musician. The father not "
        "only put no obstacles in the way of his son's pursuit of a musical "
        "career, he and his wife positively encouraged the boy."
    )
