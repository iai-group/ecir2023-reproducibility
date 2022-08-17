"""Generates new JSON file with automatic query rewrites for 2022 topics.
"""
import json
import os
from typing import Dict


def create_flat_automatic_topics_file(topics_path: str) -> None:
    """Generates new JSON file with automatic query rewrites for 2022 topics.

    Uses flattened duplicated topics file with manual rewrites provided by
    organizers and replaces manual_rewritten_utterance with
    automatic_rewritten_utterance based on file with conversation trees with
    automatic rewrites.

    Args:
        topics_path: Path to 2022 topics files.
    """
    raw_automatic_dict: Dict[str, str] = {}

    with open(
        os.path.join(
            topics_path, "2022_automatic_evaluation_topics_tree_v1.0.json"
        )
    ) as json_file:
        automatic_topics = json.load(json_file)
        for raw_topic in automatic_topics:
            for turn in raw_topic.get("turn"):
                raw_automatic_dict[turn.get("utterance")] = turn.get(
                    "automatic_rewritten_utterance"
                )

    modified_topics = []

    with open(
        os.path.join(topics_path, "2022_manual_evaluation_topics_v1.0.json")
    ) as json_file_2:
        raw_topics = json.load(json_file_2)
        for raw_topic in raw_topics:
            modified_turns = []
            for turn in raw_topic.get("turn"):
                turn["automatic_rewritten_utterance"] = raw_automatic_dict[
                    turn.get("utterance")
                ]
                del turn["manual_rewritten_utterance"]
                modified_turns.append(turn)
            modified_topic = {
                "number": raw_topic.get("number"),
                "turn": modified_turns,
            }
            modified_topics.append(modified_topic)

    json_object = json.dumps(modified_topics, indent=4, ensure_ascii=False)

    with open(
        os.path.join(topics_path, "2022_automatic_evaluation_topics_v1.0.json"),
        "w",
        encoding="utf-8",
    ) as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    create_flat_automatic_topics_file("data/topics/2022/")
