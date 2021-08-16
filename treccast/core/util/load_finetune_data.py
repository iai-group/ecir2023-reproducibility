from collections import defaultdict
from typing import List, Tuple

from treccast.core.ranking import Ranking
from treccast.core.query.sparse_query import SparseQuery


class FineTuneDataLoader:
    def __init__(
        self, file_name: str = "data/finetuning/finetune_001.tsv"
    ) -> None:
        """Loads the fine tuning data for bert ranker.

        Args:
            file_name (str, optional): Defaults to
            "data/finetuning/finetune_001.tsv". Should be copied from
            gustav1:/data/scratch/trec-cast-2021/data/finetuning
        """
        self._queries = defaultdict(list)
        self._file_name = file_name

        with open(file_name, "r") as f_in:
            for line in f_in:
                fields = line.rstrip().split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"The data format is invalid, "
                        f"{len(fields)} fields found {line}"
                    )
                utterance, passage, score = fields
                self._queries[utterance].append((passage, float(score)))

    def get_query_ranking_pairs(
        self,
    ) -> Tuple[List[SparseQuery], List[Ranking]]:
        """Returns query, ranking parallel list.

        Returns:
            List of query and corresponding rankings.
        """
        queries = []
        rankings = []
        for qid, (utterance_str, ranking_list) in enumerate(
            self._queries.items()
        ):
            utterance_query = SparseQuery(str(qid), utterance_str)
            r = Ranking(utterance_query)
            for doc_id, (doc, score) in enumerate(ranking_list):
                r.add_doc(doc_id=str(doc_id), score=score, doc_content=doc)
            queries.append(utterance_query)
            rankings.append(r)
        return queries, rankings


if __name__ == "__main__":
    # TODO: Write tests for FineTuneDataLoader
    # see https://github.com/iai-group/trec-cast-2021/issues/111
    ftdl = FineTuneDataLoader()
    query, ranking = ftdl.get_query_ranking_pairs()
    print(len(query), len(ranking))
