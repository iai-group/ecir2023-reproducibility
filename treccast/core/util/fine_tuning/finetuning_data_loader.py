import csv
from collections import defaultdict
from typing import List, Tuple

from treccast.core.ranking import Ranking
from treccast.core.base import Query


class FineTuningDataLoader:
    def __init__(
        self,
        file_name: str,
    ) -> None:
        """Loads the fine tuning data for neural (re)ranker.

        Args:
            file_name: Fine-tuning data file.
        """
        self._queries = defaultdict(list)
        self._file_name = file_name

        with open(file_name, "r") as f_in:
            spamreader = csv.reader(f_in, delimiter="\t")
            for fields in spamreader:
                if len(fields) == 5:
                    _, utterance, _, passage, score = fields
                else:
                    utterance, passage, score = fields
                # TODO: Once we switch to the new fine-tune data remove this
                # line since the data is expected to be binary.
                score = 1 if float(score) >= 1 else 0
                self._queries[utterance].append((passage, float(score)))

    def get_query_ranking_pairs(
        self,
    ) -> Tuple[List[Query], List[Ranking]]:
        """Returns query, ranking parallel list.

        Returns:
            List of query and corresponding rankings.
        """
        queries = []
        rankings = []
        for qid, (utterance_str, ranking_list) in enumerate(
            self._queries.items()
        ):
            utterance_query = Query(str(qid), utterance_str)
            r = Ranking(utterance_query)
            for doc_id, (doc, score) in enumerate(ranking_list):
                r.add_doc(doc_id=str(doc_id), score=score, doc_content=doc)
            queries.append(utterance_query)
            rankings.append(r)
        return queries, rankings


if __name__ == "__main__":
    # TODO: Write tests for FineTuneDataLoader
    # see https://github.com/iai-group/trec-cast-2021/issues/111
    ftdl = FineTuningDataLoader()
    query, ranking = ftdl.get_query_ranking_pairs()
    print(len(query), len(ranking))
