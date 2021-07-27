"""Main command line application."""

import argparse

from treccast.retrieval.first_pass_retrieval import FirstPassRetrieval
from treccast.retrieval.bm25_retrieval import BM25Retrieval
from treccast.retrieval.reranker.reranker import Reranker
from treccast.core.topic import construct_topics_from_file
from treccast.core.collection import ElasticSearchIndex
from treccast.core.query.sparse_query import SparseQuery

DEFAULT_TOPIC_INPUT_PATH = (
    "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
)
DEFAULT_REWRITE_OUTPUT_PATH = "data/queries-2020/rewrite_method.txt"
DEFAULT_RANKING_OUTPUT_PATH = "data/runs-2020/bm25.trec"


def rewrite(topics_path: str, output_path: str) -> None:
    pass


def _get_first_pass_retrieval(
    index_name: str, host_name: str
) -> FirstPassRetrieval:
    """Runs first pass retrieval.

    Args:
        index_name: Name of the Elasticsearch index.
        host_name: Host name for Elasticsearch process.

    Returns:
        The constructed class for first-pass retrieval.
            Currently only supports BM25.
    """
    # Can be expanded with more arguments
    esi = ElasticSearchIndex(index_name, hostname=host_name)
    return BM25Retrieval(esi)


def retrieval(
    topics_path: str,
    output_path: str,
    first_pass_retrieval: FirstPassRetrieval = None,
    reranker: Reranker = None,
) -> None:
    """Performs retrieval and saves the results to a TREC runfile.

    Args:
        topics_path: Path to topic input file.
        output_path: Path to output TREC runfile.
        first_pass_retrieval: First-pass
            retrieval model. Defaults to None.
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
                query = SparseQuery(query_id, question)
                ranking = first_pass_retrieval.retrieve(query)
                for rank, (doc_id, score) in enumerate(
                    ranking.fetch_topk_docs(1000)
                ):
                    f_out.write(
                        " ".join(
                            [
                                query_id,
                                "Q0",
                                doc_id,
                                str(rank + 1),
                                str(score[1]),
                                "BM25",
                            ]
                        )
                        + "\n"
                    )


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
        "-t",
        "--topics",
        type=str,
        nargs="?",
        const=DEFAULT_TOPIC_INPUT_PATH,
        help="Performs retrieval using the specified path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.rewrite:
        rewrite(args.topics, args.rewrite_output)
    if args.retrieval:
        first_pass_retrieval = _get_first_pass_retrieval(
            "ms_marco_trec_car", host_name="gustav1.ux.uis.no:9204"
        )
        retrieval(
            args.topics,
            args.output,
            first_pass_retrieval=first_pass_retrieval,
            reranker=None,
        )
