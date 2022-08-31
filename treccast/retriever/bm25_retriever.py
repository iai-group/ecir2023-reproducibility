"""BM25 retrieval using ElasticSearch."""

from collections import defaultdict
from typing import Any, Dict, List

from treccast.core.base import Query, SparseQuery
from treccast.core.collection import ElasticSearchIndex
from treccast.core.ranking import Ranking
from treccast.retriever.retriever import Retriever

_ES_query = Dict[str, Any]


class BM25Retriever(Retriever):
    def __init__(
        self,
        collection: ElasticSearchIndex,
        field: str = "body",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Initializes BM25 retrieval model based on Elasticsearch.

        Args:
            collection: ElasticSearch collection.
            field: Index field to query.
            k1: BM25 parameter (defaults to 1.2).
            b: BM25 parameter (defaults to 0.75).
        """
        self._collection = collection
        self._collection.update_similarity_parameters(k1=k1, b=b)
        self._field = field

    def simplify_query(self, query: SparseQuery) -> SparseQuery:
        """Converts weighted match queries to weighted terms.

        Args:
            query: Sparse query to simplify.

        Returns:
            Simpler version of a sparse query.
        """
        if not query.weighted_match_queries:
            return query

        tokens = defaultdict(float)
        for q, score in query.weighted_match_queries.items():
            for token in self.analyze_query(q):
                tokens[token] += score

        return SparseQuery(
            query.query_id,
            query.question,
            weighted_terms=tokens,
            weighted_match_phrases=query.weighted_match_phrases,
        )

    def retrieve(
        self, query: Query, num_results: int = 1000, source=True
    ) -> Ranking:
        """Performs retrieval.

        Args:
            query: Query instance.
            num_results: Number of documents to return (defaults to 1000).
            source: Whether to return document content (defaults to True).

        Returns:
            Document ranking.
        """
        # TODO Improve logging.
        # See https://github.com/iai-group/trec-cast-2021/issues/37

        if isinstance(query, SparseQuery):
            query = self.simplify_query(query)
            es_query = self.bool_query(
                weighted_terms=query.weighted_terms,
                weighted_phrases=query.weighted_match_phrases,
            )
        else:
            es_query = self.match_query(query.question)

        print(
            "Retrieving using query:\n",
            str(query),
            "\n",
        )

        return self._retrieve(
            query.query_id,
            query=es_query,
            num_results=num_results,
            source=source,
        )

    def _retrieve(
        self,
        query_id: str,
        query: _ES_query = None,
        num_results: int = 1000,
        source=True,
    ) -> Ranking:
        """Performs retrieval.

        Args:
            query_id: query ID.
            query: Elasticsearch query.
            num_results: Number of documents to return.
            source: Weather to include document content in the return set.

        Returns:
            Document ranking.
        """
        res = self._collection.es.search(
            body={"query": query},
            index=self._collection.index_name,
            _source=source,
            size=num_results,
        )

        return Ranking(
            query_id,
            [
                {
                    "doc_id": hit["_id"],
                    "score": hit["_score"],
                    "content": hit["_source"]["body"] if source else None,
                }
                for hit in res["hits"]["hits"]
            ],
        )

    def analyze_query(self, text: str) -> List[str]:
        """Parses text into a list of tokens which exist in the collection.

        Args:
            text: String to analyze.

        Returns:
            A list of tokens.
        """
        return [
            token["token"]
            for token in self._collection.es.indices.analyze(
                body={"text": text, "field": self._field},
                index=self._collection.index_name,
            )["tokens"]
        ]

    def match_query(self, query: str, weight: float = 1.0) -> _ES_query:
        """Simple elasticsearch match query.

        Args:
            query: Full query to use for retrieval.
            weight: Weight for scaling the scores.

        Returns:
            Elasticsearch query.
        """
        return {"match": {self._field: {"query": query, "boost": weight}}}

    def _term_query(self, term: str, weight: float = 1.0) -> _ES_query:
        """Sub-query for a single term to be used as part of a larger query.

        Args:
            term: Single term to add to the query.
            weight: Weight for scaling the term (Defaults to 1.0).

        Returns:
            Partial elasticsearch query.
        """
        return {"term": {self._field: {"value": term, "boost": weight}}}

    def _phrase_query(self, phrase: str, weight: float = 1.0) -> _ES_query:
        """Sub-query for a single phrase to be used as part of a larger query.

        Args:
            phrase: Phrase to add to the query.
            weight: Weight for scaling the phrase (Defaults to 1.0).

        Returns:
            Partial elasticsearch query.
        """
        return {
            "match_phrase": {self._field: {"query": phrase, "boost": weight}}
        }

    def bool_query(
        self,
        weighted_terms: Dict[str, float],
        weighted_phrases: Dict[str, float] = None,
        weighted_match_queries: Dict[str, float] = None,
    ) -> _ES_query:
        """Query that computes the scores of each term or phrase individually.

        Args:
            weighted_terms: Dictionary of terms with weights as key-value pair.
            weighted_phrases: Dictionary of phrases with weights as key-value
              pair (Defaults to None).
            weighted_match_queries: Dictionary of match queries with weights as
              key-value pair.

        Returns:
            Elasticsearch query to return documents based on the aggregated
              scores.
        """
        return {
            "bool": {
                "should": [
                    *[
                        self._term_query(term, weight)
                        for term, weight in (weighted_terms or {}).items()
                    ],
                    *[
                        self._phrase_query(phrase, weight)
                        for phrase, weight in (weighted_phrases or {}).items()
                    ],
                    *[
                        self.match_query(match, weight)
                        for match, weight in (
                            weighted_match_queries or {}
                        ).items()
                    ],
                ]
            }
        }


if __name__ == "__main__":
    # Example usage.
    esi = ElasticSearchIndex(
        "ms_marco_kilt_wapo_clean", hostname="localhost:9204"
    )
    bm25 = BM25Retriever(esi)
    query = Query(
        "81_1", "How do you know when your garage door opener is going bad?"
    )
    ranking = bm25.retrieve(query)
    for doc in ranking.fetch_topk_docs(5):
        print(f'{doc["doc_id"]}: {doc["score"]}')
