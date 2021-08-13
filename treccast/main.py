"""Main command line application."""

import argparse
import csv

from treccast.rewriter.rewriter import Rewriter, CachedRewriter
from treccast.retriever.retriever import Retriever
from treccast.retriever.bm25_retriever import BM25Retriever
from treccast.reranker.reranker import Reranker
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.t5_reranker import T5Reranker
from treccast.core.topic import construct_topics_from_file
from treccast.core.collection import ElasticSearchIndex
from treccast.core.query.sparse_query import SparseQuery
from treccast.core.query.preprocessing.tokenizer import SimpleTokenizer

DEFAULT_TOPIC_INPUT_PATH = (
    "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
)
DEFAULT_REWRITE_PATH = "data/rewrites/2020/1_Original.tsv"
DEFAULT_RANKING_OUTPUT_PATH = "data/runs-2020/bm25.trec"


def retrieve(
    topics_path: str,
    output_path: str,
    utterance_type: str = "raw",
    rewriter: Rewriter = None,
    retriever: Retriever = None,
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
    topics = construct_topics_from_file(topics_path)
    with open(output_path, "w") as trec_out, open(
        output_path.replace(".trec", ".tsv"), "w"
    ) as retrieval_out:
        tsv_writer = csv.writer(retrieval_out, delimeter="\t")
        tsv_writer.writerow(["query_id", "query", "passage_id", "passage"])
        for topic in topics:
            for turn in topic.turns:
                query_id = f"{topic.topic_id}_{turn.turn_id}"
                # TODO: Replace print with logging.
                # See: https://github.com/iai-group/trec-cast-2021/issues/37
                print(query_id)
                # Context is currently not used.
                question, _ = topic.get_question_and_context(
                    turn.turn_id, utterance_type
                )
                query = SparseQuery(
                    query_id, question, SimpleTokenizer if preprocess else None
                )
                if rewriter:
                    query = Rewriter.rewrite_query(query)
                ranking = retriever.retrieve(query)
                if reranker:
                    ranking = reranker.rerank(query, ranking)
                ranking.write_to_tsv_file(tsv_writer, question, k=1000)
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
    esi = ElasticSearchIndex(index_name, hostname=host_name)
    return BM25Retriever(esi, **kwargs)


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument(
        "-w",
        "--rewrite",
        action="store_true",
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
        "--rewrite_path",
        type=str,
        default=DEFAULT_REWRITE_PATH,
        help="Specifies the path for rewritten queries",
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
        choices=["bert", "t5"],
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
        "--es_index", default="ms_marco_trec_car", help="Elasticsearch index",
    )
    parser.add_argument(
        "--es_k1", default=1.2, help="Elasticsearch BM25 k1 parameter",
    )
    parser.add_argument(
        "--es_b", default=0.75, help="Elasticsearch BM25 b parameter",
    )
    parser.add_argument(
        "--utterance_type",
        type=str,
        default="raw",
        choices=["raw", "automatic", "manual"],
        help="Select the type of utterance to use.",
    )
    return parser.parse_args()


def main(args):
    """Main function that will be executed running this file.

    Args:
        args: Command-line arguments.
    """
    rewriter = None
    if args.rewrite:
        rewriter = _get_rewriter(args.rewrite_path)

    reranker = None
    if args.reranker == "bert":
        reranker = BERTReranker()
    if args.reranker == "t5":
        reranker = T5Reranker()

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
            utterance_type=args.utterance_type,
            rewriter=rewriter,
            retriever=retriever,
            preprocess=args.preprocess,
            reranker=reranker,
        )


if __name__ == "__main__":
    args = parse_args()
    print("Arguments:\n", args, "\n")
    main(args)
