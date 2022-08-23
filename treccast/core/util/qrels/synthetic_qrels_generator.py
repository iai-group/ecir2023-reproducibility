"""Synthetic qrels file generator based on canonical answers.

Currently, for each query there is only one highly relevant (rel=4) answer
generated, based on the canonical passage given for manual query rewrites.
"""
import csv

from treccast.core.topic import QueryRewrite, Topic


def generate_synthetic_qrels(year: str, output_path: str) -> None:
    topics = Topic.load_topics_from_file(year, QueryRewrite.MANUAL)
    with open(output_path, "w") as fout:
        csv_writer = csv.writer(fout, delimiter=" ")
        for topic in topics:
            for turn in topic.turns:
                if not turn.canonical_result_id:
                    print(
                        "Missing canonical answer for topic "
                        f"{topic.topic_id} turn {turn.turn_id}"
                    )
                    continue
                csv_writer.writerow(
                    [
                        topic.get_query_id(turn.turn_id),
                        "Q0",
                        turn.canonical_result_id,
                        "4",
                    ]
                )


def create_synthetic_qrels_from_provenance(output_path: str) -> None:
    """Creates qrels for 2022 topics from provenance documents."""
    topics = Topic.load_topics_from_file("2022", QueryRewrite.AUTOMATIC, False)
    qrels = set()
    with open(output_path, "w") as qrels_out:
        for topic in topics:
            for turn in topic.turns:
                if turn.provenance is not None:
                    for provenance in turn.provenance:
                        turn_id = f"{topic.topic_id}_{turn.turn_id}"
                        if (turn_id, provenance) not in qrels:
                            qrels.add((turn_id, provenance))
                            qrels_out.write(
                                " ".join(
                                    [
                                        turn_id,
                                        "0",
                                        provenance,
                                        "4",
                                    ]
                                )
                                + "\n"
                            )


if __name__ == "__main__":
    for year in ["2020", "2021"]:
        generate_synthetic_qrels(year, f"data/qrels/{year}_synthetic.txt")
    create_synthetic_qrels_from_provenance("data/qrels/2022_synthetic.txt")
