"""Main command line application."""

import argparse
import confuse
import csv

from treccast.rewriter.rewriter import Rewriter, CachedRewriter
from treccast.retriever.retriever import Retriever
from treccast.retriever.bm25_retriever import BM25Retriever
from treccast.reranker.reranker import Reranker
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.t5_reranker import T5Reranker
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.collection import ElasticSearchIndex

DEFAULT_TOPIC_INPUT_PATH = (
    "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
)
DEFAULT_REWRITE_PATH = "data/rewrites/2020/1_Original.tsv"
DEFAULT_RANKING_OUTPUT_PATH = "data/runs-2020/bm25.trec"
DEFAULT_BERT_RERANKER_PATH = "nboost/pt-bert-base-uncased-msmarco"


def retrieve(
    topics_path: str,
    output_path: str,
    utterance_type: str = "raw",
    rewriter: Rewriter = None,
    retriever: Retriever = None,
    field: str = "body",
    preprocess: bool = False,
    reranker: Reranker = None,
) -> None:
    """Performs (first-pass) retrieval and saves the results to a TREC runfile.
    First pass retrieval results are also saved in a tsv file.

    Args:
        topics_path: Path to topic input file.
        output_path: Path to output TREC runfile.
        retriever: First-pass retrieval model. Defaults to None.
        preprocess: If True use query preprocessing. Defaults to False.
        reranker: Reranker model. Defaults to None.
    """
    # TODO(IK): load year and optionally query rewrite mode from config file
    # See: https://github.com/iai-group/trec-cast-2021/issues/126
    year = "2021"
    queries = Topic.load_queries_from_file(year, QueryRewrite.MANUAL)
    with open(output_path, "w") as trec_out, open(
        output_path.replace(".trec", ".tsv"), "w"
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
            ranking = retriever.retrieve(query, field)
            if reranker:
                ranking = reranker.rerank(query, ranking)
            ranking.write_to_tsv_file(tsv_writer, query.question, k=1000)
            ranking.write_to_trec_file(trec_out, run_id="BM25", k=1000)


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
    esi = ElasticSearchIndex(index_name, hostname=host_name, timeout=30)
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
        reranker = BERTReranker(model_name=config["bert_reranker_path"].get())
    if config["reranker"].get() == "t5":
        reranker = T5Reranker()

    if config["retrieval"].get():
        retriever = _get_retriever(
            config["es_index"].get(),
            host_name=config["host_name"].get(),
            k1=config["es_k1"].get(),
            b=config["es_b"].get(),
        )
        retrieve(
            config["topics"].get(),
            config["output"].get(),
            utterance_type=config["utterance_type"].get(),
            rewriter=rewriter,
            retriever=retriever,
            field=config["es_field"].get(),
            preprocess=config["preprocess"].get(),
            reranker=reranker,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument(
        "-c", "--config-file", default="config/config_default.yaml"
    )
    args = parser.parse_args()
    config = confuse.Configuration("treccast")
    config.set_file(args.config_file)
    print("Loading config from {}:\n".format(args.config_file), config)
    main(config)
