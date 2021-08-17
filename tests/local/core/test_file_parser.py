"""Tests FileParser class from file_parser"""
import pytest

from treccast.core.util.file_parser import FileParser


MS_MARCO_PASSAGE_DATASET_GZ = "tests/data/ms_marco_passage_sample.tar.gz"
MS_MARCO_PASSAGE_DATASET = "tests/data/ms_marco_passage_sample.tsv"
MS_MARCO_PASSAGE_DATASET_TRECWEB = "tests/data/ms_marco_sample.trecweb"


@pytest.mark.parametrize(
    "filepath", [MS_MARCO_PASSAGE_DATASET_GZ, MS_MARCO_PASSAGE_DATASET]
)
def test_parse_tsv_tar_first(filepath: str) -> None:
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
def test_parse_tsv_tar_last(filepath: str) -> None:
    last_line = list(FileParser.parse(filepath))[-1]
    assert last_line == (
        "99\t(1841 - 1904) Contrary to legend, AntonÃ­n DvoÅÃ¡k (September"
        " 8, 1841 - May 1, 1904) was not born in poverty. His father was an "
        "innkeeper and butcher, as well as an amateur musician. The father not "
        "only put no obstacles in the way of his son's pursuit of a musical "
        "career, he and his wife positively encouraged the boy."
    )


def test_parse_trecweb_number_of_passages() -> None:
    passages = list(FileParser.parse(MS_MARCO_PASSAGE_DATASET_TRECWEB))
    assert len(passages) == 25


def test_parse_trecweb_first_passage() -> None:
    passage = next(FileParser.parse(MS_MARCO_PASSAGE_DATASET_TRECWEB))
    assert passage == (
        "MARCO_D1555982-0",
        "The hot glowing surfaces of stars "
        "emit energy in the form of electromagnetic radiation.?",
        "Science & Mathematics Physics The "
        "hot glowing surfaces of stars emit energy in the form of "
        "electromagnetic radiation.? It is a good approximation to assume that "
        "the emissivity e is equal to 1 for these surfaces. Find the radius of "
        "the star Rigel, the bright blue star in the constellation Orion that "
        "radiates energy at a rate of 2.7 x 10^32 W and has a surface "
        "temperature of 11,000 K. Assume that the star is spherical. Use σ =..."
        " show more Follow 3 answers Answers Relevance Rating Newest Oldest "
        "Best Answer: Stefan-Boltzmann law states that the energy flux by "
        "radiation is proportional to the forth power of the temperature: q = "
        "ε · σ · T^4 The total energy flux at a spherical surface of Radius R "
        "is Q = q·π·R² = ε·σ·T^4·π·R² Hence the radius is R = √ ( Q / "
        "(ε·σ·T^4·π) ) = √ ( 2.7x10+32 W / (1 · 5.67x10-8W/m²K^4 · (1100K)^4 · "
        "π) )",
    )


def test_parse_trecweb_last_passage() -> None:
    passages = list(FileParser.parse(MS_MARCO_PASSAGE_DATASET_TRECWEB))
    assert passages[-1] == (
        "MARCO_D301595-6",
        "Developmental Milestones and Your <8>-Year-Old Child",
        "A Word From Verywell Your 8-year-old is in the full bloom of childhood"
        ". Enjoy activities and explore the world together. It's a great time "
        "to spark new interests in your child and watch her grow in every way. "
        "Sources: Anthony, Michelle. The emotional lives of 8-10-year-olds. "
        "Scholastic Publishing. Chaplin TM, Aldao A. Gender differences in "
        "emotion expression in children: A meta-analytic review Psychological "
        "Bulletin. 2013;139 (4):735-765. doi:10.1037/a0030737. Middle childhood"
        '. CDC. "',
    )
