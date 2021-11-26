"""
Outputs a CSV file with query ID as index column and delta reciprocal rank,
recall, and NDCG@3 between automatic and manual queries as other columns.

The file is sorted by delta NDCG@3 desc.
"""
import os

import pandas as pd

from treccast.core.topic import Topic, QueryRewrite

COMMAND = (
    "tools/trec_eval/trec_eval -q -l 2 "
    "-m recip_rank -m recall.1000 -m ndcg_cut.3 data/qrels/{year}.txt "
    "/data/scratch/trec-cast-2021/data/runs/{year}/{rewrite}_1k_t5.trec"
)


def add_measures(df, year: str) -> pd.DataFrame:
    """Runs TREC eval on manual and automatic runs and calculates metrics NDCG@3,
    MRR, and recall. Additionally, adds deltas for listed measures between the
    query rewrites.

    Args:
        df: DataFrame to which to add measure columns.
        year: Year for which to run.

    Returns:
        DataFrame with manual measures and deltas for each turn.
    """
    results = os.popen(COMMAND.format(year=year, rewrite="manual"))
    for line in results:
        metric, query_id, score = line.split()
        df.loc[query_id, f"manual_{metric}"] = float(score)
        df.loc[query_id, f"delta_{metric}"] = float(score)

    results = os.popen(COMMAND.format(year=year, rewrite="automatic"))
    for line in results:
        metric, query_id, score = line.split()
        df.loc[query_id, f"delta_{metric}"] -= float(score)

    return df.iloc[(-df.delta_ndcg_cut_3).argsort()]


def add_topic_shifts(df, year: str) -> pd.DataFrame:
    """Adds topic shift annotations to DataFrame.

    Args:
        df: DataFrame to which to add topic shift annotations.
        year: Year for which to run.

    Returns:
        DataFrame with added topic shifts annotations.
    """
    if year != "2020":
        return df

    with open("data/annotations/2020_topic_shift_labels.tsv") as annotations:
        for annotation in annotations:
            query_id, _, shift = annotation.split("\t")
            df.loc[query_id, "topic_shift"] = shift

    return df


def add_query_texts(df, year: str) -> pd.DataFrame:
    """Adds all three (raw, automatic, and manual) query rewrites to DataFrame.

    Args:
        df: DataFrame to which to add query texts.
        year: Year for which to run.

    Returns:
        DataFrame with added query texts.
    """
    for rewrite in ["raw", "automatic", "manual"]:
        for query in Topic.load_queries_from_file(
            year, None if rewrite == "raw" else QueryRewrite[rewrite.upper()]
        ):
            df.loc[query.query_id, f"{rewrite}_query"] = query.question

    return df


if __name__ == "__main__":
    for year in ["2020", "2021"]:
        df = pd.DataFrame()
        df.index.name = "query_id"
        df = add_measures(df, year)
        df = add_topic_shifts(df, year)
        df = add_query_texts(df, year)
        df.to_csv(f"data/analysis/delta_metrics_{year}.csv", index=True)
