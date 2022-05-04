"""Main command line application."""

import argparse
import csv
from typing import List

import confuse

from treccast.core.collection import ElasticSearchIndex
from treccast.core.base import Query
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.ranking import CachedRanking
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.reranker import Reranker
from treccast.reranker.t5_reranker import T5Reranker
from treccast.retriever.bm25_retriever import BM25Retriever
from treccast.retriever.retriever import CachedRetriever, Retriever
from treccast.rewriter.rewriter import CachedRewriter, Rewriter

DEFAULT_CONFIG_PATH = "config/defaults/{}.yaml"


def main(config: confuse.Configuration):
    """Executes the specified configuration.

    Loads queries, initializes query rewriter, retriever, and re-ranker
    and runs the program.

    Args:
        config: Configuration generated from YAML configuration file.
    """
    query_rewrite = None
    if config["query_rewrite"].get():
        query_rewrite = QueryRewrite[config["query_rewrite"].get().upper()]
    queries = Topic.load_queries_from_file(config["year"].get(), query_rewrite)

    rewriter = None
    if config["rewrite"].get(bool):
        rewriter = _get_rewriter(config["rewrite_path"].get())

    retriever = _get_retriever(config)

    ranking_cache = None
    # Initialize CachedRanking if we are expanding candidate pools with
    # previous turns. Parameter k (candidate pool size) is updated to a
    # multiple of 1 plus the number of previous turns.
    k = config["k"].get()
    num_prev_turns = config["num_prev_turns"].get()
    if num_prev_turns:
        ranking_cache = CachedRanking(num_prev_turns, k)
        k *= num_prev_turns + 1

    reranker = _get_reranker(config)

    run(
        queries=queries,
        output_name=config["output_name"].get(),
        retriever=retriever,
        rewriter=rewriter,
        reranker=reranker,
        year=config["year"].get(),
        k=k,
        ranking_cache=ranking_cache,
    )


def run(
    queries: List[Query],
    output_name: str,
    retriever: Retriever,
    rewriter: Rewriter = None,
    reranker: Reranker = None,
    year: str = "2021",
    k: int = 1000,
    ranking_cache: CachedRanking = None,
) -> None:
    """Iterates over queries to perform rewriting, retrieval, and re-ranking.

    Results are saved to a TREC runfile. Passages are also saved in a TSV file.
    This is convenient when using later stages of the pipeline.

    Args:
        output_path: Path to output TREC runfile.
        year: Year for which to run the application.
        rewriter: Rewriter to use. Defaults to None
        retriever: First-pass retrieval model. Defaults to None.
        reranker: Reranker model. Defaults to None.
        k: number of documents to save for each turn. Defaults to 1000
        ranking_cache: Class that adds rankings from previous turns to the
            current candidate pool.
    """
    with open(f"data/runs/{year}/{output_name}.trec", "w") as trec_out, open(
        f"data/first_pass/{year}/{output_name}.tsv", "w"
    ) as retrieval_out:
        tsv_writer = csv.writer(retrieval_out, delimiter="\t")
        tsv_writer.writerow(
            ["query_id", "query", "passage_id", "passage", "label"]
        )
        for query in queries:
            # TODO: Replace print with logging.
            # See: https://github.com/iai-group/trec-cast-2021/issues/37
            print(query.query_id)

            # Custom rewriter
            if rewriter:
                query = Rewriter.rewrite_query(query)

            # Retrieval
            ranking = retriever.retrieve(query, num_results=k)
            if ranking_cache:
                ranking = ranking_cache.add_previous_turns(
                    query.get_topic_id(), ranking
                )

            # Re-ranking
            if reranker:
                ranking = reranker.rerank(query, ranking)

            # Save results
            ranking.write_to_tsv_file(tsv_writer, query.question, k=k)
            ranking.write_to_trec_file(
                trec_out,
                run_id="BM25",
                k=k,
                remove_passage_id=(year == "2021"),
            )


def _get_rewriter(path: str) -> Rewriter:
    """Returns rewriter instance that generates rewritten questions.

    Args:
        path: Filepath containing rewrites.

    Returns:
        Rewriter class containing rewrites.
    """
    return CachedRewriter(path)


def _get_retriever(config: confuse.Configuration) -> Retriever:
    """Returns (first-pass) retriever instance.

    Returns CachedRetriever (i.e., loads rankings from file) if first_pass_file
    is specified in the config.

    Args:
        config: Configuration for the run.

    Returns:
        The constructed class for first-pass retrieval.
    """
    first_pass_file = config["first_pass_file"].get()
    if first_pass_file:
        return CachedRetriever(first_pass_file)

    # Can be expanded with more arguments
    esi = ElasticSearchIndex(
        index_name=config["es"]["index_name"].get(),
        hostname=config["es"]["host_name"].get(),
        timeout=120,
    )
    return BM25Retriever(
        esi,
        field=config["es"]["field"].get(),
        k1=config["es"]["k1"].get(),
        b=config["es"]["b"].get(),
    )


def _get_reranker(config: confuse.Configuration) -> Reranker:
    """Returns re-ranker instance.

    Currently supports BERT and T5 re-rankers.

    Args:
        config: Configuration for the run.

    Raises:
        ValueError: Unsupported re-ranker.

    Returns:
        The constructed class for re-ranking.
    """
    reranker = config["reranker"].get()
    if reranker == "bert":
        return BERTReranker(
            base_model=config["bert"]["base_model"].get(),
            model_path=config["bert"]["reranker_path"].get(),
        )
    elif reranker == "t5":
        return T5Reranker()
    elif reranker:
        raise ValueError('Unsupported re-ranker. Use "bert" or "t5".')


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Args:
        args: List of arguments simulating command line arguments. It is
            useful for tests or if this file is executed programatically.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="main.py")
    # General config
    parser.add_argument(
        "-c",
        "--config-file",
        help="Path to configuration file to overwrite default values. "
        "Defaults to None",
    )
    parser.add_argument(
        "-y",
        "--year",
        choices=["2020", "2021"],
        help='Year for which to run the program. Defaults to "2021".',
    )
    parser.add_argument(
        "-k",
        help="Specifies the number of documents to retrieve at each stage. "
        "Defaults to 1000.",
    )
    parser.add_argument(
        "--num_prev_turns",
        help="Specifies the number of previous turns that should be added to the"
        " current candidate pool. Defaults to 0.",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        help='Specifies the output name for the ranking. Defaults to "raw_1k"',
    )

    # Rewriter specific config
    rewrite_group = parser.add_argument_group("Rewrite")
    rewrite_group.add_argument(
        "--query_rewrite",
        choices=["automatic", "manual"],
        help="Uses query rewrite of chosen type if specified. Defaults to None.",
    )
    rewrite_group.add_argument(
        "-w",
        "--rewrite",
        action="store_const",
        const=True,
        help="Rewrites queries if specified. Defaults to False.",
    )
    rewrite_group.add_argument(
        "--rewrite_path",
        help="Specifies the path for rewritten queries. Defalts to None.",
    )

    # First-pass retrieval specific config
    retrieval_group = parser.add_argument_group(
        "Retrieval",
        "Retrieval is always performed, either with direct querying against the "
        "index or by loading rankings from a previous run.",
    )
    retrieval_group.add_argument(
        "--es.host_name",
        dest="es.host_name",
        help='Elasticsearch host name. Defaults to "gustav1.ux.uis.no:9204".',
    )
    retrieval_group.add_argument(
        "--es.index_name",
        dest="es.index_name",
        help='Elasticsearch index name. Defaults to "ms_marco_kilt_wapo_clean".',
    )
    retrieval_group.add_argument(
        "--es.field",
        dest="es.field",
        help='Elasticsearch field to query. Defaults to "catch_all".',
    )
    retrieval_group.add_argument(
        "--es.k1",
        dest="es.k1",
        help="Elasticsearch BM25 k1 parameter. Defaults to 1.2.",
    )
    retrieval_group.add_argument(
        "--es.b",
        dest="es.b",
        help="Elasticsearch BM25 b parameter. Defaults to 0.75.",
    )

    # Reranking specific config
    reranker_group = parser.add_argument_group(
        "Retrieval",
        "Retrieval is always performed, either with direct querying against the "
        "index or by loading rankings from a previous run.",
    )
    reranker_group.add_argument(
        "--reranker",
        choices=["bert", "t5"],
        help="Performs re-ranking if specified. Defaults to None.",
    )
    reranker_group.add_argument(
        "--bert.base_model",
        dest="bert.base_model",
        help='Base bert model to use. Defaults to "bert-base-uncased".',
    )
    reranker_group.add_argument(
        "--bert.reranker_path",
        dest="bert.reranker_path",
        help="Uses fine-tuned models from the specified path. Currently "
        'applicable only to "bert" re-ranker. Defaults to None.',
    )
    return parser.parse_args(args)


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads config from config file and command line parameters.

    Loads default values from `config/defaults/config_default.yaml`. Values are
    then updated with any value specified in the command line arguments.

    Args:
        args: Arguments parsed with argparse.
    """
    # Load default config
    config = confuse.Configuration("treccast")
    config.set_file(DEFAULT_CONFIG_PATH.format("general"))

    # Load year specific config
    year = args.year or config["year"].get()
    config.set_file(DEFAULT_CONFIG_PATH.format(year))

    # Load additional config (update defaults).
    if args.config_file:
        config.set_file(args.config_file)

    # Update config from command line arguments
    config.set_args(args, dots=True)

    # Save run config to metadata file
    output_name = config["output_name"].get()
    with open(f"data/runs/{year}/{output_name}.meta.yaml", "w") as f:
        f.write(config.dump())

    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    main(config)
