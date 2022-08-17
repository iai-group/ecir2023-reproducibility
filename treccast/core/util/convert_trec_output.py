"""Converts TREC runfile to the format convinient for manual inspection.
"""
import pandas as pd
from treccast.core.util.passage_loader import PassageLoader
from treccast.core.topic import Topic, QueryRewrite


def covert_trec_file_for_manual_inspection(
    path_to_trec_file: str, top_k: int, index_name: str, year: str
) -> pd.DataFrame:
    """Converts TREC runfile to the format convinient for manual inspection.

    The resulting file contains the following columns: Query_id, Query,
    Passage_id, Passage, and Score.

    Args:
        path_to_trec_file: Path tp the TREC file to be converted.
        top_k: Number of top documents for each query to be included in the
          converted file.
        index_name: Name of the index to be sued for loading pasages.
        year: Year for which queries are to be loaded.

    Returns:
        Pandas DataFrame with the converted TREC run file.
    """
    trec_output = pd.read_csv(
        path_to_trec_file,
        delimiter=" ",
        names=["Query_id", "Q0", "Doc_id", "Rank", "Score", "Run_id"],
    )
    print(trec_output.iloc[0]["Doc_id"])
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


if __name__ == "__main__":
    results_for_manual_inspection = covert_trec_file_for_manual_inspection(
        "data/runs/2022/t5_canard_rewrites_2022.trec",
        3,
        "ms_marco_v2_kilt_wapo",
        "2022",
    )
    results_for_manual_inspection.to_csv(
        "data/runs/2022/t5_canard_rewrites_2022.csv",
        sep="\t",
        encoding="utf-8-sig",
    )
