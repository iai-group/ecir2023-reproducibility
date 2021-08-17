"""Tests Ensemble class from Ensemble"""

from treccast.core.ensemble import Ensemble


def test_ensemble_ranking_no_thresholds():
    filepaths = [
        "tests/data/ranking_sample_1.tsv",
        "tests/data/ranking_sample_2.tsv",
        "tests/data/ranking_sample_3.tsv",
    ]
    ensemble = Ensemble(filepaths=filepaths)
    rankings = ensemble.combine_rankings()
    assert len(rankings) == 10

    expected = [
        ["001", "test query 1", "001", "test passage 1"],
        ["001", "test query 1", "002", "test passage 2"],
        ["001", "test query 1", "003", "test passage 3"],
        ["001", "test query 1", "004", "test passage 4"],
        ["001", "test query 1", "005", "test passage 5"],
        ["002", "test query 2", "001", "test passage 1"],
        ["002", "test query 2", "002", "test passage 2"],
        ["002", "test query 2", "003", "test passage 3"],
        ["002", "test query 2", "004", "test passage 4"],
        ["002", "test query 2", "005", "test passage 5"],
    ]
    assert rankings == expected


def test_ensemble_ranking_thresholds():
    filepaths = [
        "tests/data/ranking_sample_1.tsv",
        "tests/data/ranking_sample_2.tsv",
        "tests/data/ranking_sample_3.tsv",
    ]
    ensemble = Ensemble(filepaths=filepaths, rank_thresholds=[1, 2, 3])
    rankings = ensemble.combine_rankings()
    # query 1: 1, 4, 3, 5, 2 query 2: 2, 3, 4, 5
    assert len(rankings) == 9

    expected = [
        ["001", "test query 1", "001", "test passage 1"],
        ["001", "test query 1", "002", "test passage 2"],
        ["001", "test query 1", "003", "test passage 3"],
        ["001", "test query 1", "004", "test passage 4"],
        ["001", "test query 1", "005", "test passage 5"],
        ["002", "test query 2", "002", "test passage 2"],
        ["002", "test query 2", "003", "test passage 3"],
        ["002", "test query 2", "004", "test passage 4"],
        ["002", "test query 2", "005", "test passage 5"],
    ]
    assert rankings == expected
