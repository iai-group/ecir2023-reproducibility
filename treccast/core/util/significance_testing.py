import argparse
import subprocess

from scipy.stats import ttest_ind


def compute_statistical_significance(
    year: str,
    runfile_1: str,
    runfile_2: str,
    trec_eval: str = "tools/trec_eval/trec_eval",
):
    """Computing statistical significance using paired t-test.

    Args:
        year: Year for which the significance test should be calculated.
        runfile_1: Path to the first runfile.
        runfile_2: Path to the second runfile.
        trec_eval: Path to trec_eval_evaluation tool.

    """
    cutoff = "500" if year == "2021" else "1000"
    path_1 = f"data/runs/{year}/{runfile_1}"
    path_2 = f"data/runs/{year}/{runfile_2}"
    metrics = [
        "recall." + cutoff,
        "map_cut." + cutoff,
        "recip_rank",
        "ndcg_cut." + cutoff,
        "ndcg_cut.3",
    ]
    file_name_1 = path_1.split("/")[-1].replace(".trec", " ")
    file_name_2 = path_2.split("/")[-1].replace(".trec", " ")
    print(f"System 1: {file_name_1}")
    print(f"System 2: {file_name_2}")

    for metric in metrics:
        proc = subprocess.Popen(
            f"{trec_eval} -m {metric} -q -l2 data/qrels/{year}.txt {path_1}",
            shell=True,
            stdout=subprocess.PIPE,
        )
        metric_file_1 = list(
            map(float, proc.communicate()[0].decode("utf-8").split()[2::3])
        )
        # print(metric_file_1[-1])

        proc = subprocess.Popen(
            f"{trec_eval} -m {metric} -q -l2 data/qrels/{year}.txt {path_2}",
            shell=True,
            stdout=subprocess.PIPE,
        )
        metric_file_2 = list(
            map(float, proc.communicate()[0].decode("utf-8").split()[2::3])
        )
        print(metric_file_2[-1])

        stats = ttest_ind(metric_file_1, metric_file_2)
        print(f"T-test for metric {metric}:")
        print(f" - t-statistics - {round(stats.statistic, 4)}")
        print(f" - p-value - {round(stats.pvalue, 4)}")


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="significance_testing.py")
    parser.add_argument(
        "--runfile_1",
        dest="runfile_1",
        type=str,
        help="Path to the first runfile.",
    )
    parser.add_argument(
        "--runfile_2",
        dest="runfile_2",
        type=str,
        help="Path to the second runfile.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2021",
        choices=["2020", "2021"],
        help="Year for which the significance test should be calculated.",
    )
    parser.add_argument(
        "--trec_eval",
        type=str,
        default="tools/trec_eval/trec_eval",
        help=(
            "Path to trec_eval evaluation tool. Defaults to"
            "tools/trec_eval/trec_eval"
        ),
    )

    return parser.parse_args()


def main(args):
    # Example usage:
    # python -m treccast.core.util.significance_testing --year 2021 \
    #  --runfile_1 input.clarke-cc_deduplicated \
    #  --runfile_2 ance/t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.trec
    compute_statistical_significance(
        year=args.year,
        runfile_1=args.runfile_1,
        runfile_2=args.runfile_2,
        trec_eval=args.trec_eval,
    )


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
