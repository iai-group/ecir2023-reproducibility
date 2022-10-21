"""Main command line application."""

import argparse
import csv
from typing import List, Tuple, Union

import confuse
import pyterrier as pt

from treccast.core.base import Query
from treccast.core.collection import ElasticSearchIndex
from treccast.core.ranking import CachedRanking, Ranking
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.util.reciprocal_rank_fusion import ReciprocalRankFusion
from treccast.expander.prf import PRF, RM3, PrfType
from treccast.reranker.reranker import Reranker
from treccast.reranker.t5_reranker import DuoT5Reranker, T5Reranker
from treccast.retriever.ance_dense_retriever import ANCEDenseRetriever
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
        query_rewrite = QueryRewrite[config["query_rewrite"].get(str).upper()]
    queries = Topic.load_queries_from_file(config["year"].get(), query_rewrite)

    rewriter = None
    reranker_rewriter = None
    if config["rewrite"].get(bool):
        rewriter = _get_rewriter(config["rewrite_path"].get())
        if config["reranker_rewrite_path"].get() is not None:
            reranker_rewriter = _get_rewriter(
                config["reranker_rewrite_path"].get()
            )

    dense_retriever = None
    rrf = None
    retriever = _get_retriever(config)
    if isinstance(retriever, tuple):
        dense_retriever = retriever[1]
        retriever = retriever[0]
        rrf = ReciprocalRankFusion(ploader=dense_retriever._passage_loader)

    expander = _get_expander(config, retriever)

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
    second_reranker = None
    second_reranker_top_k = None
    if config["duot5"].get(bool):
        second_reranker = DuoT5Reranker()
        second_reranker_top_k = config["duot5_topk"].get()

    run(
        queries=queries,
        output_name=config["output_name"].get(),
        retriever=retriever,
        rewriter=rewriter,
        reranker_rewriter=reranker_rewriter,
        expander=expander,
        reranker=reranker,
        second_reranker=second_reranker,
        second_reranker_top_k=second_reranker_top_k,
        year=config["year"].get(),
        k=k,
        ranking_cache=ranking_cache,
        dense_retriever=dense_retriever,
        rrf=rrf,
    )


def run(
    queries: List[Query],
    output_name: str,
    retriever: Retriever,
    rewriter: Rewriter = None,
    reranker_rewriter: Rewriter = None,
    expander: PRF = None,
    reranker: Reranker = None,
    second_reranker: Reranker = None,
    second_reranker_top_k: int = 50,
    year: str = "2021",
    k: int = 1000,
    ranking_cache: CachedRanking = None,
    dense_retriever: Retriever = None,
    rrf: ReciprocalRankFusion = None,
) -> None:
    """Iterates over queries to perform rewriting, retrieval, and re-ranking.

    An optional additional reranking step may also be performed on the top-k
    reranked results.

    Results are saved to a TREC runfile. Passages are also saved in a TSV file.
    This is convenient when using later stages of the pipeline.

    Args:
        queries: List of queries.
        output_path: Path to output TREC runfile.
        retriever: First-pass retrieval model.
        rewriter: Rewriter to use. Defaults to None
        reranker_rewriter: Rewriter to use for reranker (if different than the
          rewrites used for first-pass retrieval). Defaults to None.
        expander: Class to use for query expansion. Defaults to None.
        reranker: Reranker model. Defaults to None.
        second_reranker: The second reranker model to be applied after first
          reranking step in reranker.
        second_reranker_top_k: Number of top documents in the ranking to be
          reranked by the second reranker.
        year: Year for which to run the application.
        k: number of documents to save for each turn. Defaults to 1000
        ranking_cache: Class that adds rankings from previous turns to the
          current candidate pool.
        dense_retriever: Dense retriever to use. Defaults to None.
        rrf: Reciprocal Rank Fusion object for fusing rankings. Defaults to
          None.
    """
    retrieved_query_ids = []
    with open(f"data/runs/{year}/{output_name}.trec", "w") as trec_out, open(
        f"data/first_pass/{year}/{output_name}.tsv", "w"
    ) as retrieval_out:
        tsv_writer = csv.writer(retrieval_out, delimiter="\t")
        tsv_writer.writerow(
            ["query_id", "query", "passage_id", "passage", "label"]
        )
        for query in queries:
            original_query = query

            # Custom rewriter
            if rewriter:
                query = rewriter.rewrite_query(query)

            rewritten_query = query

            if expander:
                query = expander.get_expanded_query(query)

            # Retrieval
            ranking = run_retrieval(
                sparse_query=query,
                dense_query=rewritten_query,
                k=k,
                output_name=output_name,
                retriever=retriever,
                dense_retriever=dense_retriever,
                rrf=rrf,
                ranking_cache=ranking_cache,
            )

            # Re-ranking
            ranking = run_reranking(
                query=query,
                original_query=original_query,
                reranker_rewriter=reranker_rewriter,
                reranker=reranker,
                second_reranker=second_reranker,
                second_reranker_top_k=second_reranker_top_k,
                ranking=ranking,
            )

            # Save results
            ranking.write_to_tsv_file(tsv_writer, query.question, k=k)
            ranking.write_to_trec_file(
                trec_out,
                run_id="BM25",
                k=k,
                remove_passage_id=(year == "2021"),
            )

            retrieved_query_ids.append(query.query_id)


def run_retrieval(
    sparse_query: Query,
    dense_query: Query,
    k: int,
    output_name: str,
    retriever: Retriever,
    dense_retriever: Retriever,
    rrf: ReciprocalRankFusion,
    ranking_cache: CachedRanking,
) -> Ranking:
    """Runs retrieval component for a given query.

    Args:
        sparse_query: Query for sparse retriever.
        dense_query: Query for dense retriever.
        k: number of documents to save for each turn. Defaults to 1000
        output_name: Path to output TREC runfile.
        retriever: First-pass retrieval model.
        dense_retriever: Dense retriever to use. Defaults to None.
        rrf: Reciprocal Rank Fusion object for fusing rankings. Defaults to
          None.
        ranking_cache: Class that adds rankings from previous turns to the
          current candidate pool.

    Returns:
        Ranking returned by the first-pass retrieval.
    """
    if dense_retriever is not None:
        # Sparse-dense retrieval
        sparse_ranking = retriever.retrieve(sparse_query, num_results=k)
        dense_ranking = dense_retriever.retrieve(dense_query, num_results=k)
        ranking = rrf.reciprocal_rank_fusion(
            [
                (
                    output_name.split("/")[-1] + "_sparse",
                    sparse_ranking,
                ),
                (output_name.split("/")[-1] + "_dense", dense_ranking),
            ]
        )
    elif isinstance(retriever, CachedRetriever):
        sparse_query, ranking = retriever.retrieve(sparse_query, num_results=k)
    else:
        ranking = retriever.retrieve(sparse_query, num_results=k)
    if ranking_cache:
        ranking = ranking_cache.add_previous_turns(
            sparse_query.get_topic_id(), ranking
        )
    return ranking


def run_reranking(
    query: Query,
    original_query: Query,
    reranker_rewriter: Rewriter,
    reranker: Reranker,
    second_reranker: Reranker,
    second_reranker_top_k: int,
    ranking: Ranking,
) -> Ranking:
    """Runs re-ranking component for a given query.

    Args:
        query: Rewritten query used in first-pass retrieval.
        original_query: Original, raw query.
        reranker_rewriter: Rewriter to use for reranker (if different than the
          rewrites used for first-pass retrieval). Defaults to None.
        reranker: Reranker model. Defaults to None.
        second_reranker: The second reranker model to be applied after first
          reranking step in reranker.
        second_reranker_top_k: Number of top documents in the ranking to be
          reranked by the second reranker.
        ranking: Ranking returned in first-pass retrieval.

    Returns:
        Reranked ranking.
    """
    rewritten_query = query
    if reranker:
        if reranker_rewriter:
            rewritten_query = reranker_rewriter.rewrite_query(original_query)
        ranking = reranker.rerank(rewritten_query, ranking)
        if second_reranker is not None:
            ranking = second_reranker.rerank(
                rewritten_query, ranking, second_reranker_top_k
            )
    return ranking


def _get_rewriter(path: str) -> Rewriter:
    """Returns rewriter instance that generates rewritten questions.

    Args:
        path: Filepath containing rewrites.

    Returns:
        Rewriter class containing rewrites.
    """
    return CachedRewriter(path)


def _get_retriever(
    config: confuse.Configuration,
) -> Union[Retriever, Tuple[Retriever, Retriever]]:
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

    bm25_retriever = BM25Retriever(
        esi,
        field=config["es"]["field"].get(),
        k1=config["es"]["k1"].get(),
        b=config["es"]["b"].get(),
    )

    if config["ance"].get(bool):
        print("*** ANCE dense retrieval ***")
        if not pt.started():
            pt.init()
        print(config["ance_index"].get())
        ance_retriever = ANCEDenseRetriever(
            index_path=config["ance_index"].get(),
            year=config["year"].get(),
            reset_index=False,
            es_index_name=config["es"]["index_name"].get(),
            es_host_name=config["es"]["host_name"].get(),
            k=config["k"].get(),
        )
        return bm25_retriever, ance_retriever

    return bm25_retriever


def _get_expander(config: confuse.Configuration, retriever: Retriever) -> PRF:
    """Returns prf expander instance.

    Args:
        config: Configuration for the run.
        retriever: Retriever to use for document retrieval for prf.

    Returns:
        The constructed class for query expansion.
    """
    prf_type = config["prf"]["type"].get()
    if prf_type and PrfType[prf_type] == PrfType.RM3:
        return RM3(
            retriever,
            config["prf"]["num_documents"].get(),
            config["prf"]["num_terms"].get(),
        )


def _get_reranker(config: confuse.Configuration) -> Reranker:
    """Returns re-ranker instance.

    Currently supports T5 re-rankers.

    Args:
        config: Configuration for the run.

    Raises:
        ValueError: Unsupported re-ranker.

    Returns:
        The constructed class for re-ranking.
    """
    reranker = config["reranker"].get()
    if reranker == "t5":
        return T5Reranker()
    elif reranker:
        raise ValueError('Unsupported re-ranker. Use "t5".')


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
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None"
        ),
    )
    parser.add_argument(
        "-y",
        "--year",
        choices=["2020", "2021"],
        help='Year for which to run the program. Defaults to "2021".',
    )
    parser.add_argument(
        "-k",
        help=(
            "Specifies the number of documents to retrieve at each stage. "
            "Defaults to 1000."
        ),
    )
    parser.add_argument(
        "--num_prev_turns",
        help=(
            "Specifies the number of previous turns that should be added to "
            "the current candidate pool. Defaults to 0."
        ),
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
        help=(
            "Uses query rewrite of chosen type if specified. Defaults to None."
        ),
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
        help="Specifies the path for rewritten queries. Defaults to None.",
    )
    rewrite_group.add_argument(
        "--reranker_rewrite_path",
        help="Specifies the path for rewritten queries to be used for reranker"
        "(if different than the rewrites used for first-pass retrieval).\n"
        "Defaults to None.",
    )

    # First-pass retrieval specific config
    retrieval_group = parser.add_argument_group(
        "Retrieval",
        "Retrieval is always performed, either with direct querying against "
        "the index or by loading rankings from a previous run.",
    )
    retrieval_group.add_argument(
        "--es.host_name",
        dest="es.host_name",
        help='Elasticsearch host name. Defaults to "localhost:9204".',
    )
    retrieval_group.add_argument(
        "--es.index_name",
        dest="es.index_name",
        help=(
            'Elasticsearch index name. Defaults to "ms_marco_kilt_wapo_clean".'
        ),
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
    retrieval_group.add_argument(
        "--ance",
        action="store_const",
        const=True,
        help="ANCE dense retrieval.",
    )
    retrieval_group.add_argument(
        "--ance_index",
        dest="ance_index",
        type=str,
        help="Path to the ANCE dense retrieval index.",
    )

    # Reranking specific config
    reranker_group = parser.add_argument_group(
        "Reranking",
        "Reranking can be performed either using T5 model. By default the "
        "reranking module is not used.",
    )
    reranker_group.add_argument(
        "--reranker",
        choices=["t5"],
        help="Performs re-ranking if specified. Defaults to None.",
    )
    reranker_group.add_argument(
        "--duot5",
        action="store_const",
        const=True,
        help=(
            "Performs re-ranking with duoT5 after first reranking. Defaults to "
            "False."
        ),
    )
    reranker_group.add_argument(
        "--duot5_topk",
        type=int,
        dest="duot5_topk",
        help=(
            "Number of top documents in the ranking to be reranked by duoT5. "
            "Defaults to 50."
        ),
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
