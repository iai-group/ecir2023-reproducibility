"""Merges two rankings with Reciprocal Rank Fusion."""

from typing import List, Tuple

from treccast.core.ranking import Ranking
from treccast.core.util.passage_loader import PassageLoader
from trectools import TrecRun, fusion


class ReciprocalRankFusion(object):
    def __init__(
        self,
        ploader: PassageLoader,
    ) -> None:
        """Instantiates Reciprocal Rank Fusion of several rankings.

        Args:
            ploader: Passage loader.
        """
        self._ploader = ploader

    @staticmethod
    def create_trec_run(ranking: Tuple[str, Ranking]) -> TrecRun:
        """Creates trectools TrecRun object using a ranking name and a Ranking.

        Args:
            ranking: Tuple with a ranking name and a Ranking.

        Returns:
            TrecRun object.
        """
        ranking_file_path = f"data/reciprocal_rank_fusion/{ranking[0]}.trec"
        with open(ranking_file_path, "w") as trec_out:
            ranking[1].write_to_trec_file(
                trec_out,
                run_id=ranking[0],
                k=len(ranking[1].documents()[0]),
                remove_passage_id=False,
            )

        return TrecRun(ranking_file_path)

    @staticmethod
    def create_trec_runs(rankings: List[Tuple[str, Ranking]]) -> List[TrecRun]:
        """Creates trectools TrecRun objects.

        Args:
            ranking: List of tuples with a ranking name and a Ranking object.

        Returns:
            List of TrecRun objects.
        """
        return [
            ReciprocalRankFusion.create_trec_run(ranking)
            for ranking in rankings
        ]

    def reciprocal_rank_fusion(
        self, rankings: List[Tuple[str, Ranking]]
    ) -> Ranking:
        """Merges several rankings using Reciprocal Rank Fusion.

        Args:
            ranking: List of tuples with a ranking name and a Ranking object.

        Returns:
            Ranking obtained after Reciprocal Rank Fusion.
        """
        runs = ReciprocalRankFusion.create_trec_runs(rankings)
        fused_run = fusion.reciprocal_rank_fusion(runs)
        fused_run_file_path = (
            "data/reciprocal_rank_fusion/"
            f"{rankings[0][0]}_{rankings[1][0]}_fused.trec"
        )
        fused_run.print_subset(fused_run_file_path, topics=fused_run.topics())
        qid_ranking_dict = Ranking.load_rankings_from_runfile(
            fused_run_file_path, self._ploader
        )
        return list(qid_ranking_dict.values())[0]
