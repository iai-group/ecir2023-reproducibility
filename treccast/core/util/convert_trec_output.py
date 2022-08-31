"""Converts TREC runfile to the format convenient for manual inspection."""
import argparse

import pandas as pd
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.util.passage_loader import PassageLoader

_TOP_K = 3
_YEAR = "2022"
_INDEX_NAME = "ms_marco_v2_kilt_wapo_new"


def covert_trec_file_for_manual_inspection(
    path_to_trec_file: str, top_k: int, index_name: str, year: str
) -> pd.DataFrame:
    """Converts TREC runfile to the format convenient for manual inspection.

    The resulting file contains the following columns: Query_id, Query,
    Passage_id, Passage, and Score.

    Args:
        path_to_trec_file: Path to the TREC file to be converted.
        top_k: Number of top documents for each query to be included in the
          converted file.
        index_name: Name of the index to be used for loading pasages.
        year: Year for which queries are to be loaded.

    Returns:
        Pandas DataFrame with the converted TREC runfile.
    """
    trec_output = pd.read_csv(
        path_to_trec_file,
        delimiter=" ",
        names=["Query_id", "Q0", "Doc_id", "Rank", "Score", "Run_id"],
    )
    current_query_id = ""
    documents_added_for_current_query = 0
    results_for_manual_inspection = pd.DataFrame(
        columns=["Query_id", "Query", "Passage_id", "Passage", "Score"]
    )
    passage_loader = PassageLoader(index=index_name)
    queries = Topic.load_queries_from_file(year, QueryRewrite.MANUAL)
    for _, row in trec_output.iterrows():
        if current_query_id != row["Query_id"]:
            current_query_id = row["Query_id"]
            documents_added_for_current_query = 0
        if documents_added_for_current_query < top_k:
            passage = passage_loader.get(row["Doc_id"])
            query = next(
                query.question
                for query in queries
                if query.query_id == row["Query_id"]
            )
            results_for_manual_inspection = (
                results_for_manual_inspection.append(
                    {
                        "Query_id": row["Query_id"],
                        "Query": query,
                        "Passage_id": row["Doc_id"],
                        "Passage": passage,
                        "Score": row["Score"],
                    },
                    ignore_index=True,
                )
            )
            documents_added_for_current_query = (
                documents_added_for_current_query + 1
            )

    return results_for_manual_inspection


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
       Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="convert_trec_output.py")
    parser.add_argument(
        "trec_file",
        type=str,
        help="Path to the TREC file to be converted.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_TOP_K,
        help="Number of top documents for each query to include in file.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=_YEAR,
        help="Year for which queries are to be loaded.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=_INDEX_NAME,
        help="Name of the index to be used for loading pasages.",
    )
    return parser.parse_args()


def main(args):
    """Converts TREC runfile to csv file for manual inspection.

    Args:
        args: Arguments.
    """
    results_for_manual_inspection = covert_trec_file_for_manual_inspection(
        args.trec_file,
        args.k,
        args.index,
        args.year,
    )

    results_for_manual_inspection.to_csv(
        args.output_file,
        sep="\t",
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
