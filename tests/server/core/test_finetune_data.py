"""Tests the Fine-Tuning data generator"""

import os
import csv

from treccast.core.util.finetune_data import generate_finetune_data
from treccast.core.util.passage_loader import PassageLoader


def test_generate_finetune_data():
    ploader = PassageLoader(
        hostname="gustav1.ux.uis.no:9204", index="ms_marco_trec_car_clean"
    )
    finetune_filepath = "data/finetuning/finetune-test.tsv"
    generate_finetune_data(finetune_filepath, ploader)
    assert os.path.isfile(finetune_filepath)
    target_q_id = "33_1"
    target_passage_id = "CAR_02c9a9536ba5edd76b1b7a1a20b49cf628a7c87e"
    with open(finetune_filepath, "r") as f_in:
        tsv_reader = csv.reader(f_in, delimiter="\t")
        for row in tsv_reader:
            query_id, _, passage_id, passage, _ = row
            if query_id == target_q_id and passage_id == target_passage_id:
                break
    assert "\t" in passage
