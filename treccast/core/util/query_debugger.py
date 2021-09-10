"""Interactive command line tool for manually inspecting  query results.

Run the script and provide the year (2019 or 2020), which determines the Qrels,
as well as the runfile to interact with a simple prompt.

Typical usage:

python -m treccast.core.util.query_debugger --year 2020 \
  --runfile /data/scratch/trec-cast-2021/data/runs/2020/org_baselines/y2_manual_results_500.v1.0.run \
  --hostname gustav1.ux.uis.no:9204
"""

import argparse
from typing import Dict, List, Tuple

import pandas as pd
from pandas.core.frame import DataFrame
from treccast.core.qrel import Qrel
from treccast.core.ranking import Ranking
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.util.passage_loader import PassageLoader
from treccast.core.query import Query


class Debugger(object):
    def __init__(
        self,
        queries: List[Query],
        ploader: PassageLoader,
        qrel_dict: Dict[str, Qrel],
        ranking_dict: Dict[str, Ranking],
    ) -> None:
        """Organizes the interactive prompt and query results lookup.

        Initialization may take some time.

        Args:
            queries: List of Query objects loaded from topics file.
            ploader: Instantiated PassageLoader connected to Elasticsearch
            instance.
            qrel_dict: Dictionary of Qrel objects from qrels file, with query ID
            as key.
            ranking_dict: Dictionary of Ranking objects from runfile, with query
            ID as key.
        """
        super().__init__()
        self._queries = queries
        self._query_dict = {query.query_id: query for query in self._queries}
        self._ploader = ploader
        self._qrel_dict = qrel_dict
        self._ranking_dict = ranking_dict

    def repl(self):
        """Runs a read-evaluate-print loop, looks up passages given a query ID."""
        query_id = "start"
        while query_id != "":
            try:
                query_id = input("Enter Query ID> ")
            except query_id == "":
                break
            if len(query_id) > 0:
                query = self._query_dict.get(query_id)
                print(f"Query: {query.question}")
                run_df, qrels_rel_df = self.get_results(query_id)
                self.display(run_df, qrels_rel_df)

    def get_results(self, query_id: str) -> Tuple[DataFrame]:
        """Gets results from both Qrels and runfile for given query ID.

        Args:
            query_id: Query ID.

        Returns:
            A Dataframe for top-20 ranked passages, and a Dataframe for (other)
            relevant passages from qrels.
        """
        run_ranking = self._ranking_dict.get(query_id)
        run_top20 = run_ranking.fetch_topk_docs(k=20)
        qrel = self._qrel_dict.get(query_id)
        # Get only relevant passages, note that relevance label is binary.
        qrel_rel_docs = qrel.get_docs(1)
        qrel_rel_dict = {
            doc["doc_id"]: tuple([doc["content"], doc["rel"]])
            for doc in qrel_rel_docs
        }
        run_data = {
            "doc_id": [],
            "passage": [],
            "score": [],
            "relevance": [],
        }
        spent_doc_ids = []
        for doc in run_top20:
            rel = None  # Shows up as NaN in display.
            doc_id = doc["doc_id"]
            if doc_id in qrel_rel_dict:
                rel = qrel_rel_dict[doc_id][1]
                spent_doc_ids.append(doc_id)
            run_data["doc_id"].append(doc_id)
            run_data["passage"].append(doc["content"])
            run_data["score"].append(doc["score"])
            run_data["relevance"].append(rel)
        run_df = pd.DataFrame.from_dict(run_data)

        qrels_rel_data = {"doc_id": [], "passage": []}

        for doc in qrel_rel_docs:
            # Prepare to display passages labeled as relevant.
            # Skip relevant passages already retrieved in the top 20 ranking.
            # No need to display the same passages again.
            if doc["doc_id"] not in spent_doc_ids and doc["rel"]:
                qrels_rel_data["doc_id"].append(doc["doc_id"])
                qrels_rel_data["passage"].append(doc["content"])
        qrels_rel_df = pd.DataFrame.from_dict(qrels_rel_data)
        return run_df, qrels_rel_df

    def display(self, run_df: DataFrame, qrels_rel_df: DataFrame) -> None:
        """Displays the results for Query ID.

        Args:
            run_df: The top 20 runfile passages for the query.
            qrels_rel_df: The relevant passages from Qrels.
        """
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            1000,
        ):
            print("Top-20 results from runfile:")
            print(run_df)
            print("Relevant passages from QRELS:")
            print(qrels_rel_df)


def parseargs() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each CLI argument.
    """
    parser = argparse.ArgumentParser(prog="query_debugger.py")
    parser.add_argument(
        "-y",
        "--year",
        type=str,
        default="2020",
        help="Which year should be used for QRELS",
    )
    parser.add_argument(
        "-r",
        "--runfile",
        type=str,
        default="data/runs/2020/org_baselines/y2_manual_results_500.v1.0.run",
        help="Specifies the path to the runfile",
    )
    parser.add_argument(
        "-n",
        "--hostname",
        type=str,
        default="localhost:9204",  # Or: "gustav1.ux.uis.no:9204"
        help="Specifies the hostname and port to the Elasticsearch instance",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default="ms_marco_trec_car_clean",
        help="Specifies the index on the Elasticsearch instance",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        default="data/topics/2020/2020_manual_evaluation_topics_v1.0.json",
        help="Specifies the topics file to extract queries from",
    )
    parser.add_argument(
        "-q",
        "--query_rewrite",
        type=str,
        default="manual",  # or else: "automatic"
        help="Specifies the utterance rewrite approach taken from topics file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    # Instantiate PassageLoader:
    ploader = PassageLoader(args.hostname, args.index)
    # Determine query rewrite approach:
    if args.query_rewrite == "automatic":
        query_rewrite = QueryRewrite.AUTOMATIC
    elif args.query_rewrite == "manual":
        query_rewrite = QueryRewrite.MANUAL
    # Load list of Query objects from topics file:
    queries = Topic.load_queries_from_file(args.year, query_rewrite)
    # Load dict of Qrel objects from qrel file:
    qrel_dict = Qrel.load_qrels_from_file(
        f"data/qrels/{args.year}.txt", ploader
    )
    # Load dict of Ranking objects from runfile:
    ranking_dict = Ranking.load_rankings_from_runfile(args.runfile, ploader)

    # Instantiate debugger object, which will provide the REPL/prompt
    debugger = Debugger(queries, ploader, qrel_dict, ranking_dict)
    debugger.repl()
