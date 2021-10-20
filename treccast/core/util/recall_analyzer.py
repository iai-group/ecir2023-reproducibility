"""Calculates changes in recall between turns in a conversation."""
import json
from collections import defaultdict
from typing import Dict, List, Set, Union
import argparse

from treccast.core.qrel import Qrel
from treccast.core.ranking import Ranking
from treccast.core.topic import Topic
from treccast.core.util.file_parser import FileParser

DependencyList = List[Union[str, Dict[str, Union[int, str]]]]

CONFIG_RECALL = "config/analysis/recall.json"


def get_dependencies() -> Dict[str, DependencyList]:
    """Returns a dictionary of dependencies with display name and a lists of
    instructions of what retrieved document ids to use for calculating recall.
    """
    with open(CONFIG_RECALL, "r") as f:
        return json.load(f)


def calculate_recall(retrieved: Set[str], relevant: Set[str]) -> float:
    """Calculates recall based on the lists of retrieved and relevant documents.

    Args:
        retrieved: Set of retrieved documents.
        relevant: Set of ground truth documents.

    Returns:
        Recall.
    """
    if not relevant:
        return None
    return len(retrieved.intersection(relevant)) / len(relevant)


def get_retrieved_docs_for_turn(
    qid: str,
    dependency_list: DependencyList,
    rankings: Dict[str, Dict[str, Ranking]],
    cache: List[Set[str]],
) -> Set[str]:
    """Returns a set of retrieved document ids based on the dependencies in the
    dependency list.

    Args:
        qid: Query ID.
        dependency_list: List of rankings that should be added to the set of
            rankings.
        rankings: Dict of Rankings from which relevant rankings can be obtained.
        cache: List of previous turns retrieved document ID.

    Returns:
        Set of retrieved document IDs.
    """
    retrieved_doc_ids = set()
    for dependency in dependency_list:
        if type(dependency) == dict and "prev" in dependency:
            # get previous turns
            previous = cache[: dependency["prev"]].copy()

            # Update cache with all dependencies before this one
            cache.insert(0, retrieved_doc_ids.copy())

            # combine previous turns ids with current; Previous needs to be
            # flattened.
            retrieved_doc_ids.update(el for lst in previous for el in lst)

        else:
            rewrite, k = (
                (dependency, 1000)
                if type(dependency) == str
                else (dependency["rewrite"], dependency["k"])
            )
            retrieved_doc_ids.update(
                [
                    doc["doc_id"]
                    for doc in rankings[rewrite]
                    .get(qid, Ranking(""))
                    .fetch_topk_docs(k)
                ]
            )

    return retrieved_doc_ids


def get_document_appearance(
    topics: List[Topic],
    qrels: Dict[str, Qrel],
    rankings: Dict[str, Dict[str, Ranking]],
) -> Dict[str, Dict[str, List[str]]]:
    """Returns a dictionary of document IDs and a list containing in which
    document set this document appears.

    Args:
        topics: List of topics (conversations).
        qrels: Dictionary containing a list of relevant documents for each query.
        rankings: All loaded dictionaries containing a lists of retrieved
            documents for each query.

    Returns:
        Dictionary with a list of appearances for each document ID.
    """
    relevant_documents = defaultdict(dict)
    dependencies = get_dependencies()
    for topic in topics:
        previous_cache = defaultdict(list)
        for turn in topic.turns:
            qid = topic.get_query_id(turn.turn_id)
            relevant_doc_ids = set(
                doc["doc_id"] for doc in qrels.get(qid, Qrel(qid)).get_docs(1)
            )

            relevant_documents[topic.topic_id][turn.turn_id] = {
                doc_id: [] for doc_id in relevant_doc_ids
            }
            for full_name in [
                "raw",
                "raw+raw_prev_1",
                "raw+raw_prev_2",
                "raw+raw_prev_3",
                "raw+raw_prev_4",
                "raw+raw_prev_5",
                "raw+raw_prev_all",
                "automatic",
                "manual",
            ]:
                retrieved_doc_ids = get_retrieved_docs_for_turn(
                    qid,
                    dependencies[full_name],
                    rankings,
                    previous_cache[full_name],
                )
                for doc_id in relevant_documents[topic.topic_id][turn.turn_id]:
                    if doc_id in retrieved_doc_ids:
                        relevant_documents[topic.topic_id][turn.turn_id][
                            doc_id
                        ].append(full_name)

            # Add only prev
            full_name = "raw_prev_all"
            previous_docs = {
                el for lst in previous_cache[full_name] for el in lst
            }
            for doc_id in relevant_documents[topic.topic_id][turn.turn_id]:
                if doc_id in previous_docs:
                    relevant_documents[topic.topic_id][turn.turn_id][
                        doc_id
                    ].append(full_name)

            retrieved_doc_ids = [
                doc["doc_id"]
                for doc in rankings["raw"]
                .get(qid, Ranking(""))
                .fetch_topk_docs(1000)
            ]
            previous_cache[full_name].append(retrieved_doc_ids.copy())

    return relevant_documents


def get_recalls(
    topics: List[Topic],
    qrels: Dict[str, Qrel],
    rankings: Dict[str, Dict[str, Ranking]],
    topic_shifts: Dict[str, bool],
) -> Dict[int, List[float]]:
    """Returns a dictionary of topic IDs and a list containing recall for each
    turn.

    Args:
        topics: List of topics (conversations).
        qrels: Dictionary containing a list of relevant documents for each query.
        rankings: All loaded dictionaries containing a lists of retrieved
            documents for each query.

    Returns:
        Dictionary with a list of recalls for each topic.
    """
    # setup
    recalls = {
        "total_relevant_documents": 0,
        "num_valid_turns": 0,
        "num_total_turns": 0,
        "total_recalls": defaultdict(dict),
        "topics": {},
    }
    dependencies = get_dependencies()

    # iterate topics and turns
    for topic in topics:
        recalls["topics"][topic.topic_id] = {
            "num_valid_turns": 0,
            "num_total_turns": 0,
            "average_recalls": defaultdict(dict),
            "recalls": defaultdict(list),
        }
        average_recalls = recalls["topics"][topic.topic_id]["average_recalls"]
        previous_cache = defaultdict(list)
        for turn in topic.turns:
            qid = topic.get_query_id(turn.turn_id)

            # relevant doc ids
            relevant_doc_ids = set(
                doc["doc_id"] for doc in qrels.get(qid, Qrel(qid)).get_docs(1)
            )

            # basic counters
            recalls["total_relevant_documents"] += len(relevant_doc_ids)
            recalls["topics"][topic.topic_id]["num_total_turns"] += 1
            recalls["num_total_turns"] += 1
            if relevant_doc_ids:
                recalls["topics"][topic.topic_id]["num_valid_turns"] += +1
                recalls["num_valid_turns"] += +1

            # resolve dependency lists
            for full_name, dependency_list in dependencies.items():
                retrieved_doc_ids = get_retrieved_docs_for_turn(
                    qid,
                    dependency_list,
                    rankings,
                    previous_cache[full_name],
                )

                num_candidates = len(retrieved_doc_ids)
                recall = calculate_recall(retrieved_doc_ids, relevant_doc_ids)
                recalls["topics"][topic.topic_id]["recalls"][full_name].append(
                    {
                        "turn": turn.turn_id,
                        "recall": recall,
                        "num_candidates": num_candidates,
                        "topic_shift": topic_shifts.get(qid)
                        if topic_shifts
                        else None,
                    }
                )
                if not relevant_doc_ids:
                    continue

                average_recalls[full_name]["recall"] = (
                    average_recalls[full_name].get("recall", 0) + recall
                )
                average_recalls[full_name]["num_avg_candidates"] = (
                    average_recalls[full_name].get("num_avg_candidates", 0)
                    + num_candidates
                )

                recalls["total_recalls"][full_name]["recall"] = (
                    recalls["total_recalls"][full_name].get("recall", 0)
                    + recall
                )
                recalls["total_recalls"][full_name]["num_avg_candidates"] = (
                    recalls["total_recalls"][full_name].get(
                        "num_avg_candidates", 0
                    )
                    + num_candidates
                )

        # divide to get averages
        for recall_obj in average_recalls.values():
            recall_obj["recall"] /= recalls["topics"][topic.topic_id][
                "num_valid_turns"
            ]
            recall_obj["num_avg_candidates"] //= recalls["topics"][
                topic.topic_id
            ]["num_valid_turns"]

    for recall_obj in recalls["total_recalls"].values():
        recall_obj["recall"] /= recalls["num_valid_turns"]
        recall_obj["num_avg_candidates"] //= recalls["num_valid_turns"]
    return recalls


def parse_cmd_line_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year")
    args = parser.parse_args()

    args.PATH_QREL = f"data/qrels/{args.year}.txt"
    args.PATH_TOPIC_SHITFS = (
        f"data/topics/{args.year}/{args.year}_topic_shift_labels.tsv"
    )
    args.ANALYSIS_RECALL = f"data/analysis/{args.year}_recall.json"
    args.ANALYSIS_DOCUMENT_APPEARANCE = (
        f"data/analysis/{args.year}_relevant_documents.json"
    )
    args.PATH_RUNFILE = "/data/scratch/trec-cast-2021/data/runs/{}/{}_10k.trec"
    return args


if __name__ == "__main__":
    args = parse_cmd_line_args()
    topics: List[Topic] = Topic.load_topics_from_file(args.year)
    qrels: Dict[str, Qrel] = Qrel.load_qrels_from_file(args.PATH_QREL)
    rankings: Dict[str, Dict[str, Ranking]] = {
        name: Ranking.load_rankings_from_runfile(
            args.PATH_RUNFILE.format(args.year, name)
        )
        for name in ["raw", "automatic", "manual"]
    }
    topic_shifts = (
        {
            qid: bool(int(shift))
            for qid, _, shift in (
                line.split("\t")
                for line in FileParser.parse(args.PATH_TOPIC_SHITFS)
            )
        }
        if args.year == "2020"
        else None
    )

    recalls = get_recalls(topics, qrels, rankings, topic_shifts)
    with open(args.ANALYSIS_RECALL, "w") as f:
        f.write(json.dumps(recalls))

    relevant_documents = get_document_appearance(topics, qrels, rankings)
    with open(args.ANALYSIS_DOCUMENT_APPEARANCE, "w") as f:
        f.write(json.dumps(relevant_documents))
