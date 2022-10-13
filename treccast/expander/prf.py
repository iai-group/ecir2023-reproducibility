"""Classes for query expansion by pseudo-relevance-feedback."""

import abc
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, List

from treccast.core.base import Query, SparseQuery
from treccast.retriever.bm25_retriever import BM25Retriever

_WeightedTerms = Dict[str, float]

_LAMBDA = 0.5


class PrfType(Enum):
    RM3 = "RM3"


class PRF(abc.ABC):
    """Abstract class for pseudo relevance feedback."""

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    @abc.abstractmethod
    def get_expanded_query(self, query: Query) -> SparseQuery:
        """Expands given query with additional terms."""
        raise NotImplementedError

    def interpolate_terms(
        self,
        weighted_terms: _WeightedTerms,
        weighted_terms_to_add: _WeightedTerms,
        lam: float = _LAMBDA,
    ) -> Dict[str, float]:
        """Interpolates new weighted terms into the existing terms.

        Args:
            weighted_terms: Original terms.
            weighted_terms_to_add: Terms to interpolate.
            lam: Weight ratio between old and new terms. If <0.5, new terms will
              be rated higher than old ones.
        """
        terms = dict.fromkeys(
            weighted_terms.keys() | weighted_terms_to_add.keys(), 0
        )
        for term in terms:
            terms[term] = lam * weighted_terms.get(term, 0) + (
                1 - lam
            ) * weighted_terms_to_add.get(term, 0)
        return terms


class RM3(PRF):
    def __init__(
        self,
        retriever: BM25Retriever,
        prf_num_documents: int = 10,
        prf_num_terms: int = 10,
    ) -> None:
        """Pseudo relevance feedback based on RM3 algorithm.

        The algorithm follows
          https://dl.acm.org/doi/pdf/10.1145/3130348.3130376.

        Args:
            retriever: Retriever to use to get top relevant documents.
            prf_num_documents: Number of retrieved documents to use for prf
              (defaults to 10).
            prf_num_terms: Number of top scoring terms to use for prf
              (defaults to 10).
        """
        super().__init__(retriever)
        self.prf_num_documents = prf_num_documents
        self.prf_num_terms = prf_num_terms

    def get_expanded_query(self, query: Query) -> SparseQuery:
        """Returns expanded sparse query.

        Args:
            query: Query to use for the initial query retrieval.

        Returns:
            Sparse query containing expanded list of weighted terms.
        """
        if isinstance(query, SparseQuery):
            query_terms = self.retriever.simplify_query(query).weighted_terms
        else:
            query_terms = Counter(self.retriever.analyze_query(query.question))
        rm3_terms = self._get_top_collection_terms(query)
        return SparseQuery(
            query.query_id,
            query.question,
            self.interpolate_terms(query_terms, rm3_terms),
        )

    def _get_top_collection_terms(self, query: Query) -> Dict[str, float]:
        """Returns top terms and weights associated with each term.

        Number of documents to consider and number of terms to take are
        specified in self.num_documents and self.num_terms respectively.

        Args:
            query: Query for the initial retrieval.

        Returns:
            A dictionary with weighted terms according to the RM3 algorithm.
        """
        fbWeights = defaultdict(float)
        top_ranked_documents: List[Dict[str, Any]] = self.retriever.retrieve(
            query,
            num_results=self.prf_num_documents,
            source=False,
        ).fetch_topk_docs(self.prf_num_documents)

        for doc in top_ranked_documents:
            tv: Dict[str, Any] = self.retriever._collection.es.termvectors(
                index=self.retriever._collection.index_name,
                id=doc.doc_id,
                fields=self.retriever._field,
                field_statistics=False,
                offsets=False,
                payloads=False,
                positions=False,
                term_statistics=False,
            )["term_vectors"][self.retriever._field]["terms"]
            doc_length = sum(stats["term_freq"] for stats in tv.values())
            for term, payload in tv.items():
                fbWeights[term] += (
                    payload["term_freq"] / doc_length * doc.score
                )

        sorted_query_terms = sorted(
            fbWeights.items(), key=lambda item: item[1], reverse=True
        )[: self.prf_num_terms]
        total_weights = sum(fbWeight for _, fbWeight in sorted_query_terms)

        return {
            term: fbWeight / total_weights
            for term, fbWeight in sorted_query_terms
        }
