"""Create a json topic file with expanded queries from mixed initiative."""
import argparse

import csv
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)


def generate_2022_topics_mi(
    original_topics_filepath: str, output_filepath: str, mi_query_filepath: str
) -> None:
    """Generates 2022 topic JSON file with mixed initiative query.

    Args:
        original_topics_filepath: Original topic JSON filepath.
        output_filepath: Output JSON topic filepath.
        mi_query_filepath: Path to the mixed initiative queries (TSV file).
    """
    # Loads original topic JSON file.
    with open(original_topics_filepath, "r", encoding="utf8") as f_in:
        original_topics = json.load(f_in)

    # Loads mixed initiative queries.
    with open(mi_query_filepath, "r", encoding="utf8") as f_mi:
        tsv_reader = csv.reader(f_mi, delimiter="\t")
        mi_queries = {}
        for row in tsv_reader:
            query_id, turn_leaf_id, utterance = row[:3]
            mi_queries[f"{query_id}|{turn_leaf_id}"] = utterance

    # Adds mixed initiative queries.
    for topic_idx, topic in enumerate(original_topics):
        turn_leaf_id = f"{topic['number']}_{topic['turn'][-1]['number']}"
        for turn_idx, turn in enumerate(topic["turn"]):
            mi_query_id = f"{topic['number']}_{turn['number']}|{turn_leaf_id}"
            if mi_query_id not in mi_queries:
                logging.info(
                    "Missing mixed initiative query "
                    f"{topic['number']}_{turn['number']}"
                )
            original_topics[topic_idx]["turn"][turn_idx][
                "mi_expanded_utterance"
            ] = mi_queries[mi_query_id]
            original_topics[topic_idx]["turn"][turn_idx][
                "turn_leaf_id"
            ] = turn_leaf_id

    # Dumps topics to JSON file.
    with open(output_filepath, "w", encoding="utf8") as f_out:
        json.dump(original_topics, f_out, indent=4)


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
       Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="create_2022_evaluation_topics_mi.py")
    parser.add_argument(
        "original_topics",
        type=str,
        help="Path to the original topic JSON file.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output JSON topic file.",
    )
    parser.add_argument(
        "mi_query_file",
        type=str,
        help="Path to the mixed initiative queries (TSV file).",
    )
    return parser.parse_args()


def main(args):
    """Create a JSON topic file with expanded queries from mixed initiative.

    Args:
        args: Arguments.
    """
    generate_2022_topics_mi(
        args.original_topics,
        args.output_file,
        args.mi_query_file,
    )


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
