"""Converts TREC mixed-initiative runfile to the JSON format for submission.

Each response corresponds to one subtree. A ranking for a given turn_id in one
subtree contains one text field, one rank field and multiple provenances.

Exemplary JSON runfile:
{
    "run_name": "uis_duoboat",
    "run_type": "automatic_mi",
    "turns": [
        {
            "turn_id": "132_1-1",
            "responses": [
                {
                    "text": "HOME - UN Climate Change Conference (COP26) ...",
                    "rank": 1,
                    "provenance": [
                        {
                            "id": "MARCO_30_1605964568-1",
                            "text": "HOME - UN Climate Change Conference ...",
                            "score": 411.6572314041887
                        },
                        {
                            "id": "MARCO_30_1605964568-1",
                            "text": "HOME - UN Climate Change Conference ...",
                            "score": 411.6572314041887
                        },
                        ...
                    ]
                },
                {
                    "text": "HOME - UN Climate Change Conference (COP26) ...",
                    "rank": 2,
                    "provenance": [
                        {
                            "id": "MARCO_30_1605964568-1",
                            "text": "HOME - UN Climate Change Conference ...",
                            "score": 411.6572314041887
                        },
                        {
                            "id": "MARCO_30_1605964568-1",
                            "text": "HOME - UN Climate Change Conference ...",
                            "score": 411.6572314041887
                        },
                        ...
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

import pandas as pd
from treccast.core.util.passage_loader import PassageLoader


def convert_trec_file_to_json_file(
    trec_file_path: str,
    output_json_file_path: str,
    run_name: str,
    index_name: str,
) -> None:
    """Converts TREC mixed-initiative runfile to the JSON format for submission.

    Args:
        trec_file_path: The path to the TREC runfile.
        output_json_file_path: The output path for the runfile in JSON format.
        run_name: The name of the run.
        index_name: The name of the index to be used for loading passages.
    """
    trec_output = pd.read_csv(
        trec_file_path,
        delimiter=" ",
        names=["Query_id", "Q0", "Doc_id", "Rank", "Score", "Run_id"],
    )

    turns = []
    responses = []

    passage_loader = PassageLoader(index=index_name)
    how_many_branches = trec_output.groupby("Query_id").Q0.count()
    current_turn = trec_output.Query_id.iloc[0]
    responses = []
    rank = 1  # for responses
    for (qid, _), group in trec_output.groupby(["Query_id", "Q0"]):
        if current_turn != qid:
            turns.append({"turn_id": current_turn, "responses": responses})
            responses = []
            rank = 1  # for responses
            current_turn = qid
        # how many branches are there -- how many provenances can we submit
        n_branches = how_many_branches[qid] / 1000
        max_provenance_per_response = 1000 / n_branches

        response = {"provenance": [], "rank": rank}
        for _, row in group[
            group["Rank"] < max_provenance_per_response
        ].iterrows():
            loaded_text = passage_loader.get(row["Doc_id"])
            if row["Rank"] == 1:
                response["text"] = loaded_text

            response["provenance"].append(
                {
                    "id": row["Doc_id"],
                    "text": loaded_text,
                    "score": row["Score"],
                }
            )
        responses.append(response)
        rank += 1

    turns.append({"turn_id": current_turn, "responses": responses})

    with open(
        output_json_file_path, "w", encoding="utf-8"
    ) as json_submission_file:
        json.dump(
            {
                "run_name": run_name,
                "run_type": "automatic_mi",
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
    parser = argparse.ArgumentParser(prog="generate_MI_submission_file.py")
    # General config
    parser.add_argument(
        "--trec_file",
        help="The path to the TREC runfile.",
    )
    parser.add_argument(
        "--output_file",
        help="The output path for the runfile in JSON format.",
    )
    parser.add_argument(
        "--run_name",
        help="The name of the run.",
    )
    parser.add_argument(
        "--index_name",
        default="ms_marco_v2_kilt_wapo_new",
        help="The name of the index to be used for loading passages.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    convert_trec_file_to_json_file(
        trec_file_path=args.trec_file,
        output_json_file_path=args.output_file,
        run_name=args.run_name,
        index_name=args.index_name,
    )
