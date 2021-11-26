"""Generates the inner part of a latex table. Dictionary keys in NOTATION
contain the file endings of files to include as rows in the latex table.

The columns are standard trec measures (Recall, MRR, MAP, and NDCG@3). Measures
for both 2020 and 2021 are output side by side in a single table.
"""

import os

import pandas as pd

YEARS = ["2020", "2021"]
NOTATION = {
    "raw": "R",
    "automatic": "A",
    "manual": "M",
    "_1k": "",
    "_2k": "^{**}",
    "_prev_1": "_{n,n-1}",
    "_prev_2": "_{n,...,n-2}",
    "_prev_3": "_{n,...,n-3}",
    "_prev_4": "_{n,...,n-4}",
    "_prev_5": "_{n,...,n-5}",
    "_prev_all": "_{n,...,1}",
    "_prev_1_first": "_{n,n-1,1}",
}
COMMAND = (
    "tools/trec_eval/trec_eval -l 2 "
    "-m map -m recip_rank -m recall.1000 -m ndcg_cut.3 data/qrels/{year}.txt "
    "/data/scratch/trec-cast-2021/data/runs/{year}/{rewrite}{run_details}_t5.trec"
)


def run_eval() -> pd.DataFrame:
    """Runs evaluation and generates pandas dataframe for each relevant metric.

    Returns:
        Dataframe where each row is a different run.
    """
    df = pd.DataFrame(
        columns=[
            "rewrite",
            "Run",
            "Recall_2020",
            "MRR_2020",
            "MAP_2020",
            "NDCG@3_2020",
            "Recall_2021",
            "MRR_2021",
            "MAP_2021",
            "NDCG@3_2021",
        ]
    )

    for rewrite in ["raw", "automatic", "manual"]:
        for run_details in [
            "_1k",
            "_prev_1",
            "_prev_2",
            "_prev_3",
            "_prev_4",
            "_prev_5",
            "_prev_all",
            "_prev_1_first",
            "_2k",
        ]:
            row = [rewrite, f"${NOTATION[rewrite]}{NOTATION[run_details]}$"]
            for year in YEARS:
                scores = {
                    "recall_1000": 0,
                    "recip_rank": 0,
                    "map": 0,
                    "ndcg_cut_3": 0,
                }
                results = os.popen(
                    COMMAND.format(
                        year=year,
                        rewrite=rewrite,
                        run_details=run_details,
                    )
                )
                for line in results:
                    metric, _, score = line.split()
                    if metric in scores:
                        scores[metric] = float(score)
                row.extend(scores.values())
            df.loc[len(df)] = row
    return df


def add_bold(series: pd.Series) -> pd.Series:
    """Adds latex bold markup for highest scores in the table.

    Args:
        series: Series for which to find max scores.
    """
    if series.name not in ["rewrite", "Run"]:
        max_val = f"{series.max():.3f}"
        series[:] = [f"{val:.3f}" for val in series]
        series[series == max_val] = f"\\textbf{{{max_val}}}"
    return series


def save_table_to_file(df: pd.DataFrame) -> None:
    """Saves table to file. Note that it only generates inner rows of a latex
    table and not the entire environment.

    Args:
        df: Dataframe which contains table rows information.
    """
    current_run = "raw"
    with open("results.txt", "w") as f:
        for _, row in df.iterrows():
            if current_run != row[0]:
                current_run = row[0]
                f.write("\\hline\n")
            f.write("\t&\t".join(row.values[1:]) + "\\\\\n")


if __name__ == "__main__":
    df = run_eval()
    df = (
        df.groupby("rewrite")
        .apply(lambda group: group.apply(lambda x: add_bold(x), axis=0))
        .astype("str")
    )
    save_table_to_file(df)
