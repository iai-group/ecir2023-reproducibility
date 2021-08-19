"""This module parses the Wizard of Wikipedia data and generates fine-tuning
 data for fine-tuning rerankers."""

import csv
import json
from random import randint
from typing import Dict, List, Tuple


def get_query_sentence_pairs(json_obj: Dict) -> List[Tuple[str]]:
    """Parses a json object containing dialogs and passages and returns query,
    doc, score tuples.

    Args:
        json_obj: JSON object containing Wizard of Wikipedia data.

    Returns:
        List of query, doc, score tuples from the given data.
    """
    query_sentence_pairs = []
    for dialogue in json_obj:
        turns = dialogue["dialog"]
        for i, turn in enumerate(turns):
            if not (
                "wizard" in turn["speaker"].lower() and turn["checked_passage"]
            ):
                continue

            key, rel_passage = list(turn["checked_passage"].items())[0]
            if "partner" not in key:
                continue

            prev_turn = turns[i - 1]
            retrieved_passages = [
                list(d.keys())[0].replace("&amp;", "&")
                for d in prev_turn["retrieved_passages"]
            ]
            retrieved_texts = [
                " ".join(list(passage.values())[0])
                for passage in prev_turn["retrieved_passages"]
            ]

            if (
                rel_passage not in retrieved_passages
                or len(retrieved_passages) < 3
            ):
                continue

            index = retrieved_passages.index(rel_passage)

            query_sentence_pairs.append(
                (prev_turn["text"], retrieved_texts.pop(index), 1)
            )
            retrieved_passages.pop(index)

            # Add non relevant train data
            for i in range(3):
                index = randint(0, len(retrieved_passages) - 1)
                query_sentence_pairs.append(
                    (prev_turn["text"], retrieved_texts.pop(index), 0)
                )
                retrieved_passages.pop(index)
    return query_sentence_pairs


def write_query_sentence_pairs(data: List[Tuple[str]], file_path: str) -> None:
    """Writes the query, sentence pairs into a file after assinging unique int
    ids.


    Args:
        data: List of query, doc tuples and score.
        file_path: Path where to store the data.
    """
    query_ids = {}
    doc_ids = {}
    qid_idx = 0
    doc_id_idx = 0
    with open(file_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="\t")
        for query, doc, score in data:
            if query in query_ids:
                qid = query_ids[query]
            else:
                qid = qid_idx
                query_ids[query] = qid_idx
                qid_idx += 1
            if doc in doc_ids:
                doc_id = doc_ids[doc]
            else:
                doc_id = doc_id_idx
                doc_ids[doc] = doc_id_idx
                doc_id_idx += 1
            csv_writer.writerow([qid, query, doc_id, doc, float(score)])


def parse_and_write_wow_data(input_file: str, output_file: str) -> None:
    """Parse the json fine and write the tsv file for the query, doc pairs from
    Wizard of Wikipedia data.

    Args:
        input_file: Json input file path.
        output_file: Output tsv file path.
    """
    with open(input_file, "r") as f:
        json_obj = json.load(f)
    train_data = get_query_sentence_pairs(json_obj)
    write_query_sentence_pairs(train_data, output_file)


if __name__ == "__main__":
    TRAIN = "data/finetuning/wizard_of_wikipedia/train.json"
    TRAIN_OUT = "data/finetuning/wizard_of_wikipedia/wow_finetune_train.tsv"
    VAL = "data/finetuning/wizard_of_wikipedia/valid_random_split.json"
    VAL_OUT = "data/finetuning/wizard_of_wikipedia/wow_finetune_val.tsv"
    TEST = "data/finetuning/wizard_of_wikipedia/test_random_split.json"
    TEST_OUT = "data/finetuning/wizard_of_wikipedia/wow_finetune_test.tsv"
    parse_and_write_wow_data(TRAIN, TRAIN_OUT)
    parse_and_write_wow_data(VAL, VAL_OUT)
    parse_and_write_wow_data(TEST, TEST_OUT)
