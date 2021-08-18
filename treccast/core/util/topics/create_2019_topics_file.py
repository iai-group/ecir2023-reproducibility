"""Generates a topic JSON file for 2019 with manual answers included (which,
unlike in 2020 and 2021, are not included in the official topic file).
"""
import csv
import json


def generate_2019_topics_file(
    original_json_file: str,
    manual_rewrites_tsv_file: str,
    output_json_file: str,
) -> None:
    """Generates 2019 topic JSON file with manual query rewrites included.

    Args:
        original_json_file: Original topic JSON file (provided by organizers).
        manual_rewrites_tsv_file: TSV file with manual query rewrites (provided
            by organizers).
        output_json_file: Output JSON file.
    """
    # Loads original topic JSON file.
    with open(original_json_file, "r", encoding="utf8") as f_in:
        raw_topics = json.load(f_in)

    # Loads manual query rewrites provided in TSV format.
    with open(manual_rewrites_tsv_file, "r", encoding="utf8") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        manual_rewrites = {}
        for row in tsv_reader:
            query_id, utterance = row[:2]
            manual_rewrites[query_id] = utterance

    # Adds manual rewrites to each turn.
    for topic_idx, topic in enumerate(raw_topics):
        for turn_idx, turn in enumerate(topic["turn"]):
            query_id = f"{topic['number']}_{turn['number']}"
            if query_id not in manual_rewrites:
                print(
                    "Missing manual rewrite for query "
                    f"{topic['number']}_{turn['number']}"
                )
            raw_topics[topic_idx]["turn"][turn_idx][
                "manual_rewritten_utterance"
            ] = manual_rewrites[query_id]

    # Dumps enriched topics to JSON file.
    with open(output_json_file, "w", encoding="utf8") as outfile:
        json.dump(raw_topics, outfile, indent=4)


if __name__ == "__main__":
    generate_2019_topics_file(
        "data/topics/2019/evaluation_topics_v1.0.json",
        "data/topics/2019/evaluation_topics_annotated_resolved_v1.0.tsv",
        "data/topics/2019/2019_manual_evaluation_topics_v1.0.json",
    )
