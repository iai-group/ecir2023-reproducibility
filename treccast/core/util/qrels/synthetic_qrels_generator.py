"""Synthetic qrels file generator based on canonical answers.

Currently, for each query there is only one highly relevant (rel=4) answer
generated, based on the canonical passage given for manual query rewrites.
"""
import csv

from treccast.core.topic import Topic, QueryRewrite


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


if __name__ == "__main__":
    for year in ["2020", "2021"]:
        generate_synthetic_qrels(year, f"data/qrels/{year}_synthetic.txt")
