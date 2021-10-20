"""Main command line application."""

import argparse
import csv
from collections import defaultdict

import confuse

from treccast.core.collection import ElasticSearchIndex
from treccast.core.ranking import Ranking
from treccast.core.topic import QueryRewrite, Topic
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.reranker import Reranker
from treccast.reranker.t5_reranker import T5Reranker
from treccast.retriever.bm25_retriever import BM25Retriever
from treccast.retriever.retriever import Retriever
from treccast.rewriter.rewriter import CachedRewriter, Rewriter

DEFAULT_CONFIG_PATH = "config/{}.yaml"


def retrieve(
    output_name: str,
    year: str = "2021",
    query_rewrite: str = "manual",
    rewriter: Rewriter = None,
    retriever: Retriever = None,
    es_field: str = "body",
    reranker: Reranker = None,
    first_pass_file: str = None,
    k: int = 1000,
    num_prev_turns: int = 0,
) -> None:
    """Performs (first-pass) retrieval and saves the results to a TREC runfile.
    First pass retrieval results are also saved in a tsv file.

    Args:
        output_path: Path to output TREC runfile.
        year: Year for which to run the application.
        query_rewrite: Type of query to use. Defaults to "manual".
        rewriter: Rewriter to use. Defaults to None
        retriever: First-pass retrieval model. Defaults to None.
        es_field: Elasticsearch field to query. Year 2021 has options title,
            body and catch_all.
        reranker: Reranker model. Defaults to None.
    """
    queries = Topic.load_queries_from_file(
        year, QueryRewrite[query_rewrite.upper()] if query_rewrite else None
    )

    rankings = (
        Ranking.load_rankings_from_tsv_file(first_pass_file)
        if first_pass_file
        else None
    )

    ranking_cache = defaultdict(list)
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

            if rewriter:
                query = Rewriter.rewrite_query(query)

            if rankings:
                # FIXME https://github.com/iai-group/trec-cast-2021/issues/228
                # NB! TSV files do not contain score so we cannot fetch top k
                # here
                ranking = rankings[query.query_id]
            else:
                ranking = retriever.retrieve(query, es_field, num_results=k)
                if num_prev_turns:
                    topic_id = query.query_id.split("_")[0]
                    ranking_cache[topic_id].append(ranking.fetch_topk_docs(k))
                    for rank in ranking_cache[topic_id][
                        -num_prev_turns - 1 : -1
                    ]:
                        ranking.update(rank)
                    print("Num docs:", len(ranking))

            if reranker:
                ranking = reranker.rerank(query, ranking)
            num_to_save = 100000 if num_prev_turns else k
            ranking.write_to_tsv_file(tsv_writer, query.question, k=num_to_save)
            ranking.write_to_trec_file(
                trec_out,
                run_id="BM25",
                k=num_to_save,
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


def _get_retriever(index_name: str, host_name: str, **kwargs) -> Retriever:
    """Returns (first-pass) retriever instance.

    Args:
        index_name: Name of the Elasticsearch index.
        host_name: Host name for Elasticsearch process.
        **kwargs: Keyword arguments. For example, parameters `b` and `k1` for
            the BM25 ranking function.

    Returns:
        The constructed class for first-pass retrieval.
            Currently only supports BM25.
    """
    # Can be expanded with more arguments
    esi = ElasticSearchIndex(index_name, hostname=host_name, timeout=120)
    return BM25Retriever(esi, **kwargs)


def main(config):
    """Main function that will be executed running this file.

    Args:
        config: Configuration generated from YAML configuration file.
    """
    rewriter = None
    if config["rewrite"].get(bool):
        rewriter = _get_rewriter(config["rewrite_path"].get())

    reranker = None
    if config["reranker"].get() == "bert":
        reranker = BERTReranker(
            base_model=config["base_bert_model"].get(),
            model_path=config["bert_reranker_path"].get(),
        )
    elif config["reranker"].get() == "t5":
        reranker = T5Reranker()

    if config["retrieval"].get():
        retriever = _get_retriever(
            config["es_index"].get(),
            host_name=config["host_name"].get(),
            k1=config["es_k1"].get(),
            b=config["es_b"].get(),
        )
        retrieve(
            output_name=config["output_name"].get(),
            year=config["year"].get(),
            query_rewrite=config["query_rewrite"].get(),
            rewriter=rewriter,
            retriever=retriever,
            es_field=config["es_field"].get(),
            reranker=reranker,
            first_pass_file=config["first_pass_file"].get(),
            k=config["k"].get(),
            num_prev_turns=config["num_prev_turns"].get(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("-c", "--config-file")
    parser.add_argument("-y", "--year")
    args = parser.parse_args()
    config = confuse.Configuration("treccast")
    config.set_file(DEFAULT_CONFIG_PATH.format("defaults/config_default"))
    config.set_file(DEFAULT_CONFIG_PATH.format(f"defaults/{args.year}"))
    config.set_file(args.config_file)
    print("Loading config from {}:\n".format(args.config_file), config)

    config["year"] = args.year
    main(config)
