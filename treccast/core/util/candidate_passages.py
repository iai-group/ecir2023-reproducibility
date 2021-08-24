"""Extracts candidate passages from canonical documents.
For each query, it collects all passages from all documents that are provided as
canonical responses in the topic file.

Note: this is meant to be used with the 2021 dataset, as there are differences
in the formatting from older versions."""

import csv

from treccast.core.topic import Topic, QueryRewrite
from treccast.core.util.passage_loader import PassageLoader


def write_candidate_passages(output_path, year):
    """Writes candidate passages and other passages in the same document to a
    tsv file.

    Args:
        output_path: Output path for the tsv file
        year: Which dataset to use
    """

    passage_loader = PassageLoader()
    topics = Topic.load_topics_from_file(year, QueryRewrite.MANUAL)
    with open(output_path, "w") as f_out:
        tsv_writer = csv.writer(f_out, delimiter="\t")
        for topic in topics:
            for i, turn in enumerate(topic.turns):
                query_id = topic.get_query_id(turn.turn_id)
                print(query_id)
                print(turn.passage_id)
                for prev_turn in topic.turns[:i]:
                    doc_id_base = prev_turn.canonical_result_id + "_"
                    doc_ids = doc_id_base.join([str(x) for x in range(i, 201)])
                    docs = passage_loader.mget(doc_ids)
                    for j, doc in enumerate(docs):
                        if not doc:
                            break
                        tsv_writer.writerow(
                            [
                                query_id,
                                turn.manual_rewritten_utterance,
                                doc_ids[j],
                                docs[j],
                            ]
                        )


if __name__ == "__main__":
    write_candidate_passages(
        "data/first_pass/2021/canonical_passages_extended.tsv", "2021"
    )
