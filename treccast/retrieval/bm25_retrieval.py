"""BM25 retrieval using ElasticSearch."""

from treccast.core.collection import ElasticSearchIndex
from treccast.core.query.sparse_query import SparseQuery
from treccast.core.ranking import Ranking
from treccast.retrieval.first_pass_retrieval import FirstPassRetrieval


class BM25Retrieval(FirstPassRetrieval):
    def __init__(self, es_collection: ElasticSearchIndex) -> None:
        """Initializes BM25 retrieval model based on Elasticsearch.

        Args:
            es_collection: ElasticSearch collection.
        """
        super().__init__(es_collection)
        self._es = es_collection.es
        self._index_name = es_collection.index_name

    def retrieve(self, query: SparseQuery, num_results: int = 1000) -> Ranking:
        """Performs retrieval.

        Args:
            query: Sparse query instance.
            num_results: Number of documents to return (defaults
                to 1000).

        Returns:
            Document ranking.
        """
        res = self._es.search(
            index=self._index_name,
            q=query.query_text,
            _source=False,
            size=num_results,
        )
        ranking = Ranking(query.query_id)
        for hit in res["hits"]["hits"]:
            ranking.add_doc(hit["_id"], None, hit["_score"])
        return ranking


if __name__ == "__main__":
    # Example usage.
    esi = ElasticSearchIndex("ms_marco", hostname="localhost:9204")
    bm25 = BM25Retrieval(esi)
    query = SparseQuery(
        "81_1", "How do you know when your garage door opener is going bad?"
    )
    ranking = bm25.retrieve(query)
    for doc_id, score in ranking.fetch_topk_docs(10):
        print(f"{doc_id}: {score}")
