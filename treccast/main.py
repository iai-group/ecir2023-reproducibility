"""Main command line application."""

from typing import List


def retrieval(topics: List[Topic]) -> None:

    for topic in topics:
        for turn_id in range(1, topic.num_turns() + 1):
            pass
            # question, context = topic.get_turn(turn_id)
            # query = get_query(question, context)
            # initial_ranking = do_first_pass_retrieval(query)
            # query2 = rewrite_query(query, initial_ranking)
            # final_ranking = rerank(initial_ranking, query2)


if __name__ == "__main__":
    # TODO: add argparse
    # Different modes:
    topics = ...  # List[Topic]
    # - query rewriting (input: TREC CAsT topic file, output: query rewrites file)
    # - retrieval using queries from file
