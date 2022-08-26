"""Converts TREC runfile to the JSON format for submission.

Exemplary JSON runfile:
{
    "run_name": "uis_duoboat",
    "run_type": "automatic",
    "turns": [
        {
            "turn_id": "132_1-1",
            "responses": [
                {
                    "text": "HOME - UN Climate Change Conference (COP26) at ...",
                    "rank": 1,
                    "provenance": [
                        {
                            "id": "MARCO_30_1605964568-1",
                            "text": "HOME - UN Climate Change Conference ...",
                            "score": 411.6572314041887
                        }
                    ]
                },
                ...
            ]
        }
    ]
}
"""

import argparse
import json
from enum import Enum

import pandas as pd
from treccast.core.util.passage_loader import PassageLoader


class RunType(Enum):
    AUTOMATIC = "automatic"
    AUTOMATIC_MI = "automatic_mi"
    MANUAL = "manual"


def convert_trec_file_to_json_file(
    trec_file_path: str,
    output_json_file_path: str,
    run_name: str,
    run_type: str,
    index_name: str,
) -> None:
    """Converts TREC runfile to the JSON format for submission.

    Args:
        trec_file_path: The path to the TREC runfile.
        output_json_file_path: The output path for the runfile in JSON format.
        run_name: The name of the run.
        run_type: The type of the run (automatic/automatic_mi/manual)
        index_name: The name of the index to be used for loading passages.
    """
    trec_output = pd.read_csv(
        trec_file_path,
        delimiter=" ",
        names=["Query_id", "Q0", "Doc_id", "Rank", "Score", "Run_id"],
    )

    turn_ids = trec_output["Query_id"].unique()
    turns = []
    current_turn_id = turn_ids[0]
    responses = []

    passage_loader = PassageLoader(index=index_name)
    for _, row in trec_output.iterrows():
        if current_turn_id is not row["Query_id"]:
            turns.append({"turn_id": current_turn_id, "responses": responses})
            responses = []
            current_turn_id = row["Query_id"]
        loaded_text = passage_loader.get(row["Doc_id"])
        responses.append(
            {
                "text": loaded_text,
                "rank": row["Rank"],
                "provenance": [
                    {
                        "id": row["Doc_id"],
                        "text": loaded_text,
                        "score": row["Score"],
                    }
                ],
            }
        )

    with open(
        output_json_file_path, "w", encoding="utf-8"
    ) as json_submission_file:
        json.dump(
            {
                "run_name": run_name,
                "run_type": run_type.value,
                "turns": turns,
            },
            json_submission_file,
            indent=4,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="generate_submission_file.py")
    # General config
    parser.add_argument(
        "--trec_file",
        help=("The path to the TREC runfile."),
    )
    parser.add_argument(
        "--output_file",
        help=("The output path for the runfile in JSON format."),
    )
    parser.add_argument(
        "--run_name",
        help=("The name of the run."),
    )
    parser.add_argument(
        "--run_type",
        default=RunType.AUTOMATIC,
        help=("The type of the run (automatic/automatic_mi/manual)."),
    )
    parser.add_argument(
        "--index_name",
        default="ms_marco_v2_kilt_wapo_new",
        help=("The name of the index to be used for loading passages."),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    convert_trec_file_to_json_file(
        trec_file_path=args.trec_file,
        output_json_file_path=args.output_file,
        run_name=args.run_name,
        run_type=args.run_type,
        index_name=args.index_name,
    )
