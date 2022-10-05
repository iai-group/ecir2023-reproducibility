import os
from typing import Callable, List
from unittest import mock

import confuse
import pytest
from treccast.core.base import Query, ScoredDocument

with pytest.helpers.mock_expensive_imports():
    from treccast import main
    from treccast.core.ranking import Ranking


class MockBM25Retriever(mock.MagicMock):
    def __init__(self, *args, **kwargs) -> None:
        """Mock BM25Retriever"""
        super().__init__(*args, **kwargs)
        self.documents = {
            "001": [
                ScoredDocument("001", "test passage", 1),
                ScoredDocument("003", "test passage", 3),
                ScoredDocument("002", "test passage", 2),
            ],
            "002": [
                ScoredDocument("002", "test passage", 2),
                ScoredDocument("005", "test passage", 5),
                ScoredDocument("004", "test passage", 4),
            ],
        }

    def retrieve(self, query: Query, num_results: int) -> Ranking:
        return Ranking(
            query.query_id, self.documents[query.query_id][:num_results]
        )


@pytest.fixture
def mock_retriever() -> MockBM25Retriever:
    with mock.patch(
        "treccast.main.BM25Retriever", new_callable=MockBM25Retriever
    ) as mock_bm25:
        yield mock_bm25


@pytest.fixture
def queries() -> List[Query]:
    return [Query("001", "test query 1"), Query("002", "test query 2")]


@pytest.fixture(params=["2020", "2021"])
def default_config(request) -> confuse.Configuration:
    args = main.parse_args(["-y", request.param, "-o", "test"])
    yield main.load_config(args)
    os.remove(f"data/runs/{request.param}/test.meta.yaml")


@pytest.mark.parametrize(
    "year, key, value",
    [
        ("2020", "field", "body"),
        ("2020", "index_name", "ms_marco_trec_car_clean"),
        ("2021", "field", "catch_all"),
        ("2021", "index_name", "ms_marco_kilt_wapo_clean"),
    ],
)
def test_load_config_default_year(year, key, value):
    args = main.parse_args(["-y", year, "-o", "test"])
    config = main.load_config(args)
    assert config["es"][key].get() == value


def test_main_load_retriever(
    mock_retriever: MockBM25Retriever, default_config: confuse.Configuration
):
    with mock.patch("treccast.main.run"):
        main.main(default_config)

    mock_retriever.assert_called_with(
        mock.ANY, field=default_config["es"]["field"].get(), k1=1.2, b=0.75
    )


@mock.patch("treccast.main.Topic.load_queries_from_file")
@mock.patch("treccast.main.run")
def test_main(
    mock_run: Callable,
    mock_load_queries: Callable,
    mock_retriever: MockBM25Retriever,
    default_config: confuse.Configuration,
    queries: List[Query],
):
    mock_load_queries.return_value = queries
    main.main(default_config)

    mock_run.assert_called_with(
        queries=queries,
        output_name="test",
        retriever=mock_retriever(),
        rewriter=None,
        expander=None,
        reranker=None,
        second_reranker=None,
        second_reranker_top_k=None,
        year=default_config["year"].get(),
        k=1000,
        ranking_cache=None,
        dense_retriever=None,
        rrf=None,
        mixed_initiative=False,
    )


# The following 4 tests test run function in main and saving results to file.
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_save_to_file_setup(
    mock_open: mock.MagicMock,
    mock_retriever: MockBM25Retriever,
    queries: List[Query],
    default_config: confuse.Configuration,
):
    year = default_config["year"].get()
    main.run(
        queries=queries,
        output_name="test",
        retriever=mock_retriever(),
        year=year,
        k=3,
    )

    # Tests opening 2 files (TREC and TSV).
    assert mock_open.call_args_list == [
        mock.call(f"data/runs/{year}/test.trec", "w"),
        mock.call(f"data/first_pass/{year}/test.tsv", "w"),
    ]


@mock.patch("treccast.core.ranking.Ranking.write_to_tsv_file")
def test_run_write_to_tsv_file(
    write_tsv: mock.MagicMock,
    mock_retriever: MockBM25Retriever,
    queries: List[Query],
    default_config: confuse.Configuration,
):
    year = default_config["year"].get()
    with mock.patch("builtins.open", mock.mock_open()):
        main.run(
            queries=queries,
            output_name="test",
            retriever=mock_retriever(),
            year=year,
            k=2,
        )

    # Tests writing to TSV file was called twice.
    assert write_tsv.call_args_list == [
        mock.call(mock.ANY, "test query 1", k=2),
        mock.call(mock.ANY, "test query 2", k=2),
    ]


@mock.patch("treccast.core.ranking.Ranking.write_to_trec_file")
def test_run_write_to_trec_file(
    write_trec: mock.MagicMock,
    mock_retriever: MockBM25Retriever,
    queries: List[Query],
    default_config: confuse.Configuration,
):
    year = default_config["year"].get()
    with mock.patch("builtins.open", mock.mock_open()):
        main.run(
            queries=queries,
            output_name="test",
            retriever=mock_retriever(),
            year=year,
            k=2,
        )

    # Tests writing to TREC file was called twice.
    assert write_trec.call_args_list == [
        mock.call(
            mock.ANY,
            k=2,
            remove_passage_id=year == "2021",
            run_id="BM25",
            leaf_id=None,
        ),
        mock.call(
            mock.ANY,
            k=2,
            remove_passage_id=year == "2021",
            run_id="BM25",
            leaf_id=None,
        ),
    ]


@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_save_to_files(
    mock_open: mock.MagicMock,
    mock_retriever: MockBM25Retriever,
    queries: List[Query],
    default_config: confuse.Configuration,
):
    year = default_config["year"].get()
    main.run(
        queries=queries,
        output_name="test",
        retriever=mock_retriever(),
        year=year,
        k=2,
    )

    # Expectation of writing to TSV and TREC files.
    # First call is writing header to TSV file. Following calls are writing to
    # TSV and TREC files two sets of two calls each. We use two queries
    # (001, 002) with k=2.
    assert mock_open().write.call_args_list == [
        mock.call("query_id\tquery\tpassage_id\tpassage\tlabel\r\n"),
        mock.call("001\ttest query 1\t003\ttest passage\t3\r\n"),
        mock.call("001\ttest query 1\t001\ttest passage\t1\r\n"),
        mock.call("001 Q0 003 1 3 BM25\n"),
        mock.call("001 Q0 001 2 1 BM25\n"),
        mock.call("002\ttest query 2\t005\ttest passage\t5\r\n"),
        mock.call("002\ttest query 2\t002\ttest passage\t2\r\n"),
        mock.call("002 Q0 005 1 5 BM25\n"),
        mock.call("002 Q0 002 2 2 BM25\n"),
    ]
