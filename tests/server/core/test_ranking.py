"""Tests Ranking class from ranking.py on server."""
from treccast.core.ranking import Ranking
from treccast.core.util.passage_loader import PassageLoader


def test_load_rankings_from_runfile_ploader():
    ploader = PassageLoader(
        hostname="gustav1.ux.uis.no:9204", index="ms_marco_trec_car_clean"
    )
    runfile = "data/runs/2020/org_baselines/y2_manual_results_500.v1.0.run"
    rankings = Ranking.load_rankings_from_runfile(runfile, ploader)
    ranking = rankings["81_1"]
    assert ranking.query_id == "81_1"
    assert len(ranking) == 500
