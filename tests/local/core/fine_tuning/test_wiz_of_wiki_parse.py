import csv
import os

from treccast.core.util.fine_tuning.wiz_of_wiki_parse import (
    parse_and_write_wow_data,
)


def test_wiz_of_wiki_data_parse():
    SAMPLE = "tests/data/wiz_of_wiki_sample.json"
    SAMPLE_OUT = "tests/data/wiz_of_wiki_sample.tsv"
    parse_and_write_wow_data(SAMPLE, SAMPLE_OUT)
    with open(SAMPLE_OUT) as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        rows = []
        for row in csvreader:
            assert len(row) == 5
            rows.append(row)
    assert rows[0][0] == "0"
    assert (
        rows[0][1]
        == "I agree. One of my favorite forms of science fiction is anything related to time travel! I find it fascinating."
    )
    assert rows[0][2] == "0"
    assert (
        "Time travel is a common theme in fiction and has been depicted in a variety of media, such as literature, television, film, and advertisements."
        in rows[0][3]
    )
    assert rows[0][4] == "1.0"
    os.remove(SAMPLE_OUT)
