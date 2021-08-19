"""BM25 retrieval using ElasticSearch."""

from treccast.core.collection import ElasticSearchIndex
from treccast.core.query import Query
from treccast.core.ranking import Ranking
from treccast.retriever.retriever import Retriever


class BM25Retriever(Retriever):
    def __init__(
        self,
        es_collection: ElasticSearchIndex,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Initializes BM25 retrieval model based on Elasticsearch.

        Args:
            es_collection: ElasticSearch collection.
            k1: BM25 parameter (defaults to 1.2).
            b: BM25 parameter (defaults to 0.75).
        """
        super().__init__(es_collection)
        self._es = es_collection.es
        self._index_name = es_collection.index_name
        es_collection.update_similarity_parameters(k1=k1, b=b)

    def retrieve(
        self, query: Query, field: str = "body", num_results: int = 1000
    ) -> Ranking:
        """Performs retrieval.

        Args:
            query: Query instance.
            field: Index field to query.
            num_results: Number of documents to return (defaults
                to 1000).

        Returns:
            Document ranking.
        """
        # TODO Improve logging.
        # See https://github.com/iai-group/trec-cast-2021/issues/37

        print(
            "Retrieving using query:\n",
            " ".join(
                token["token"]
                for token in self._es.indices.analyze(
                    body={"text": query.question}, index=self._index_name
                )["tokens"]
            ),
        )

        res = self._es.search(
            body={"query": {"match": {field: {"query": query.question}}}},
            index=self._index_name,
            _source=True,
            size=num_results,
        )

        return Ranking(
            query.query_id,
            [
                {
                    "doc_id": hit["_id"],
                    "score": hit["_score"],
                    "content": hit["_source"]["body"],
                }
                for hit in res["hits"]["hits"]
            ],
        )


if __name__ == "__main__":
    # Example usage.
    esi = ElasticSearchIndex("ms_marco", hostname="localhost:9204")
    bm25 = BM25Retriever(esi)
    query = Query(
        "81_1", "How do you know when your garage door opener is going bad?"
    )
    ranking = bm25.retrieve(query)
    for doc_id, score in ranking.fetch_topk_docs(10):
        print(f"{doc_id}: {score}")
