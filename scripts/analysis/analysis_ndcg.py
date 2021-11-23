"""
Outputs a CSV file with query ID as index column and delta reciprocal rank,
recall, and NDCG@3 between automatic and manual queries as other columns.

The file is sorted by delta NDCG@3 desc.
"""
import os

import pandas as pd

COMMAND = (
    "tools/trec_eval/trec_eval -q -l 2 "
    "-m recip_rank -m recall.1000 -m ndcg_cut.3 data/qrels/{year}.txt "
    "/data/scratch/trec-cast-2021/data/runs/{year}/{rewrite}_1k_t5.trec"
)


def get_deltas(year: str) -> pd.DataFrame:
    """Runs TREC eval on manual and automatic runs and calculates deltas per
    query for metrics NDCG@3, MRR, and recall.

    Args:
        year: year for which to run.

    Returns:
        Dataframe with deltas for each turn.
    """
    df = pd.DataFrame()
    df.index.name = "query_id"
    results = os.popen(COMMAND.format(year=year, rewrite="manual"))
    for line in results:
        metric, query_id, score = line.split()
        df.loc[query_id, metric] = float(score)

    results = os.popen(COMMAND.format(year=year, rewrite="automatic"))
    for line in results:
        metric, query_id, score = line.split()
        df.loc[query_id, metric] -= float(score)

    return df.iloc[(-df.ndcg_cut_3.abs()).argsort()]


if __name__ == "__main__":
    for year in ["2020", "2021"]:
        df: pd.DataFrame = get_deltas(year)
        df.to_csv(f"data/analysis/delta_metrics_{year}.csv", index=True)
