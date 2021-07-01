"""Main command line application."""

import argparse

DEFAULT_TOPIC_INPUT_PATH = (
    "data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json"
)
DEFAULT_TOPIC_OUTPUT_PATH = (
    "data/topics-2020/rewritten_evaluation_topics_annotated_v1.1.json"
)


def rewrite(topics_path: str, output_path: str) -> None:
    pass


def retrieval(topics_path: str) -> None:

    # for topic in topics:
    # for turn_id in range(1, topic.num_turns() + 1):
    # pass
    # question, context = topic.get_turn(turn_id)
    # query = get_query(question, context)
    # initial_ranking = do_first_pass_retrieval(query)
    # query2 = rewrite_query(query, initial_ranking)
    # final_ranking = rerank(initial_ranking, query2)
    pass


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        argparse.Namespace: Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="%(prog)s [-h] [-w [TOPIC_INPUT]] [-o [REWRITE_OUTPUT]] "
        "[-r [TOPIC_INPUT]]",
    )
    parser.add_argument(
        "-w",
        "--rewrite",
        type=str,
        nargs="?",
        const=DEFAULT_TOPIC_INPUT_PATH,
        help="Rewrites queries from the input file path, to the output path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        const=DEFAULT_TOPIC_OUTPUT_PATH,
        help="Specifies the output path for query rewriting",
    )
    parser.add_argument(
        "-r",
        "--retrieval",
        type=str,
        nargs="?",
        const=DEFAULT_TOPIC_INPUT_PATH,
        help="Performs retrieval on using the specified path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(type(args))
    if args.rewrite:
        rewrite(args.rewrite, args.output)
    if args.retrieval:
        retrieval(args.retrieval)
