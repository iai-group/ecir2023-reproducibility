"""Generate fine-tuning data using TREC CAST Y1&Y2 data."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, Tuple

from treccast.core.qrel import Qrel
from treccast.core.base import Query
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.util.passage_loader import PassageLoader


def get_query_and_qrel_dicts(
    query_rewrite: QueryRewrite = QueryRewrite.MANUAL,
    ploader: PassageLoader = None,
) -> Tuple[Dict[str, Query], Dict[str, Qrel]]:
    """Create dictionaries of Query and Qrel objects.

    Args:
        query_rewrite (optional): The type of QueryRewrite . Defaults to
        QueryRewrite.MANUAL.

    Returns:
        Two dictionaries, both with query ID as key, and with Query and Qrel
        objects as values, respectively.
    """

    if ploader is None:
        ploader = PassageLoader(
            hostname="gustav1.ux.uis.no:9204", index="ms_marco_trec_car_clean"
        )
    queries = []
    qrels_dict = {}
    for year in ["2019"]:
        queries.extend(Topic.load_queries_from_file(year, query_rewrite))
        filepath = f"data/qrels/{year}.txt"
        qrels_dict.update(Qrel.load_qrels_from_file(filepath, ploader))
    return {query.query_id: query for query in queries}, qrels_dict


def generate_finetuning_data_cast_y1y2(
    output_path: str,
    ploader: PassageLoader = None,
    query_rewrite: QueryRewrite = QueryRewrite.MANUAL,
) -> None:
    """Generate training data from qrels files, topic files, indexed passages.

    Args:
        output_path: The name of the .tsv file to output.
        ploader (optional): The PassageLoader to retrieve passages from.
        Defaults to None.
        query_rewrite (optional): The type of query utterance to load in Query
        objects. Defaults to QueryRewrite.MANUAL.
    """
    query_dict, qrel_dict = get_query_and_qrel_dicts(query_rewrite, ploader)
    counter = defaultdict(int)
    with open(output_path, "w") as f_out:
        tsv_writer = csv.writer(f_out, delimiter="\t")
        for query_id, qrel in qrel_dict.items():
            query_utterance = query_dict[query_id].question
            for doc in qrel.get_docs():
                passage_id = doc["doc_id"]
                passage = doc["content"]
                rel = doc["rel"]
                counter[rel] += 1
                tsv_writer.writerow(
                    [query_id, query_utterance, passage_id, passage, rel]
                )
    print("The scores occurred in the following frequencies:")
    for score, count in counter.items():
        print(f"{score}: {count}")


if __name__ == "__main__":
    # Generate fine-tuning data and write to file (on g1):
    ploader = PassageLoader()
    finetune_filepath = "data/fine_tuning/trec_cast/Y1_manual_qrels.tsv"
    generate_finetuning_data_cast_y1y2(finetune_filepath, ploader)
    print("Finished writing fine-tuning data to file.")
