import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re


def rank_bm25_parameters(results_directory_path: str, year: str):
    """Ranks BM25 parameters with respect to Recall@100 and saves it to file.

    Args:
        results_directory_path: Path to directory with the trec_eval results
          stored in .csv files.
        year: The year for which the ranking will be created.
    """
    all_results_df = pd.DataFrame(columns=["year", "b", "k1", "recall_1000"])
    for filename in os.listdir(
        os.path.join(os.getcwd(), results_directory_path)
    ):
        if year in filename and "bm25" not in filename:
            result_filename_path = os.path.join(
                results_directory_path, filename
            )
            results_df = pd.read_csv(
                result_filename_path,
                sep=r"\s+",
                names=["Metric", "Topics", "Values"],
            )
            recall_1000 = results_df.loc[results_df["Metric"] == "recall_1000"][
                "Values"
            ]
            file_parameters = re.findall(r"\d+\.\d+", filename)

            all_results_df = all_results_df.append(
                {
                    "year": year,
                    "b": float(file_parameters[0]),
                    "k1": float(file_parameters[1]),
                    "recall_1000": float(recall_1000),
                },
                ignore_index=True,
            )
            all_results_df.sort_values(
                ["recall_1000", "year"], ascending=[False, False], index=False
            ).to_csv(
                os.path.join(
                    os.getcwd(),
                    results_directory_path,
                    f"bm25_parameters_ranking_{year}.csv",
                ),
                sep="\t",
                mode="w+",
            )


def convert_ranking_to_pivot_dataframe(
    path_to_ranking_file: str,
) -> pd.DataFrame:
    """Converts ranking of BM25 parameters to the DataFrame.

    Args:
        path_to_ranking_file: Path to directory with the parameters ranking.

    Returns:
        Pandas DataFrame with BM25 parameters.
    """
    ranking_df = pd.read_csv(
        os.path.join(os.getcwd(), path_to_ranking_file),
        sep=r"\s+",
    )
    ranking_df = ranking_df.sort_values(
        ["b", "k1"], ascending=[True, True]
    ).drop(["year"], axis=1)
    ranking_df_pivot = ranking_df.pivot(
        index="b", columns="k1", values="recall_1000"
    )
    return ranking_df_pivot


def plot_df_on_heatmap(dataframe: pd.DataFrame, path_to_save_heatmap: str):
    """Creates heatmap from DataFrame and saves it to file.

    Args:
        dataframe: DataFrame to be used for heatmap generation.
        path_to_save_heatmap: Path to directory where the heatmap is saved.
    """
    plt.figure(figsize=(15, 10))
    heatmap = plt.pcolor(dataframe, cmap="RdBu")
    plt.yticks(np.arange(len(dataframe)), labels=dataframe.index)
    plt.xticks(
        np.arange(len(dataframe.columns)),
        labels=dataframe.columns,
    )

    for i in range(len(dataframe)):
        for j, column in enumerate(dataframe.columns):
            plt.text(
                j + 0.5,
                i + 0.5,
                "%.4f" % dataframe.iloc[i][column],
                horizontalalignment="center",
                verticalalignment="center",
            )

    plt.colorbar(heatmap)
    plt.savefig(path_to_save_heatmap.replace(".csv", ".png"))


def plot_results_on_heatmap(path_to_ranking_file: str):
    """Creates heatmap of BM25 parameters.

    Args:
        path_to_ranking_file: Path to directory with the parameters ranking.
    """
    ranking_df_pivot = convert_ranking_to_pivot_dataframe(path_to_ranking_file)
    plot_df_on_heatmap(
        ranking_df_pivot, path_to_ranking_file.replace(".csv", ".png")
    )


def plot_averaged_results_on_heatmap(
    path_to_ranking_file_1: str,
    path_to_ranking_file_2: str,
    path_to_save_heatmap: str,
):
    """Creates heatmap of two averaged BM25 parameters rankings.

    Args:
        path_to_ranking_file_1: Path to directory with the first parameters
          ranking.
        path_to_ranking_file_2: Path to directory with the second parameters
          ranking.
        path_to_save_heatmap: Path to directory where the heatmap is saved.
    """
    ranking_df_pivot_1 = convert_ranking_to_pivot_dataframe(
        path_to_ranking_file_1
    )
    ranking_df_pivot_2 = convert_ranking_to_pivot_dataframe(
        path_to_ranking_file_2
    )

    average_results_df = ranking_df_pivot_1.add(
        ranking_df_pivot_2, fill_value=0
    ).div(2)
    plot_df_on_heatmap(average_results_df, path_to_save_heatmap)


if __name__ == "__main__":
    rank_bm25_parameters("data/fine_tuning/bm25/", "2020")
    plot_results_on_heatmap(
        "data/fine_tuning/bm25/bm25_parameters_ranking_2021.csv"
    )
    if False:  # Set to true for plotting average results
        plot_averaged_results_on_heatmap(
            "data/fine_tuning/bm25/bm25_parameters_ranking_2020.csv",
            "data/fine_tuning/bm25/bm25_parameters_ranking_2021.csv",
            "data/fine_tuning/bm25/bm25_parameters_ranking_2020_2021_avg.csv",
        )
