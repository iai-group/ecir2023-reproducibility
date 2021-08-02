"""Main command line application."""

import argparse

from treccast.retriever.retriever import Retriever
from treccast.retriever.bm25_retriever import BM25Retriever
from treccast.reranker.reranker import NeuralReranker, Reranker
from treccast.core.topic import construct_topics_from_file
from treccast.core.collection import ElasticSearchIndex
from treccast.core.query.sparse_query import SparseQuery
from treccast.core.query.preprocessing.tokenizer import SimpleTokenizer

DEFAULT_TOPIC_INPUT_PATH = (
    "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
)
DEFAULT_REWRITE_OUTPUT_PATH = "data/queries-2020/rewrite_method.txt"
DEFAULT_RANKING_OUTPUT_PATH = "data/runs-2020/bm25.trec"


def rewrite(topics_path: str, output_path: str) -> None:
    pass


def _get_retriever(index_name: str, host_name: str, **kwargs) -> Retriever:
    """Returns (first-pass) retriever instance.

    Args:
        index_name: Name of the Elasticsearch index.
        host_name: Host name for Elasticsearch process.

    Returns:
        The constructed class for first-pass retrieval.
            Currently only supports BM25.
    """
    # Can be expanded with more arguments
    esi = ElasticSearchIndex(index_name, hostname=host_name)
    return BM25Retriever(esi, **kwargs)


def retrieve(
    topics_path: str,
    output_path: str,
    retriever: Retriever = None,
    preprocess: bool = False,
    reranker: Reranker = None,
) -> None:
    """Performs (first-pass) retrieval and saves the results to a TREC runfile.

    Args:
        topics_path: Path to topic input file.
        output_path: Path to output TREC runfile.
        retriever: First-pass retrieval model. Defaults to None.
        preprocess: If True use query preprocessing. Defaults to False.
        reranker: Reranker model. Defaults to None.
    """
    topics = construct_topics_from_file(topics_path)
    with open(output_path, "w") as f_out:
        for topic in topics:
            for turn in topic.turns:
                query_id = f"{topic.topic_id}_{turn.turn_id}"
                # TODO: Replace print with logging.
                # See: https://github.com/iai-group/trec-cast-2021/issues/37
                print(query_id)
                # Context is currently not used.
                question, _ = topic.get_question_and_context(turn.turn_id)
                query = SparseQuery(
                    query_id, question, SimpleTokenizer if preprocess else None
                )
                ranking = retriever.retrieve(query)
                if reranker:
                    ranking = reranker.rerank(query, ranking)
                ranking.write_to_file(f_out, run_id="BM25", k=1000)


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument(
        "-w",
        "--rewrite",
        type=bool,
        help="Rewrites queries if specified",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        const=DEFAULT_RANKING_OUTPUT_PATH,
        help="Specifies the output path for the final ranking",
    )
    parser.add_argument(
        "-p",
        "--rewrite_output",
        type=str,
        nargs="?",
        const=DEFAULT_REWRITE_OUTPUT_PATH,
        help="Specifies the output path for rewritten queries",
    )
    parser.add_argument(
        "-r",
        "--retrieval",
        type=bool,
        nargs="?",
        default=False,
        const=True,
        help="Performs retrieval if specified",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Use custom query preprocessing",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        nargs="?",
        const="nboost/pt-bert-base-uncased-msmarco",
        help="Performs reranking if specified",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        nargs="?",
        const=DEFAULT_TOPIC_INPUT_PATH,
        help="Performs retrieval using the specified path",
    )
    parser.add_argument(
        "--es.index",
        metavar="index",
        dest="es_index",
        default="ms_marco_trec_car",
        help="Elasticsearch index",
    )
    parser.add_argument(
        "--es.k1",
        metavar="k1",
        dest="es_k1",
        default=1.2,
        help="Elasticsearch BM25 k1 parameter",
    )
    parser.add_argument(
        "--es.b",
        metavar="b",
        dest="es_b",
        default=0.75,
        help="Elasticsearch BM25 b parameter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.rewrite:
        rewrite(args.topics, args.rewrite_output)
    if args.retrieval:
        retriever = _get_retriever(
            args.es_index,
            host_name="localhost:9204",
            k1=args.es_k1,
            b=args.es_b,
        )
        retrieve(
            args.topics,
            args.output,
            retriever=retriever,
            preprocess=args.preprocess,
            reranker=NeuralReranker(args.reranker) if args.reranker else None,
        )
