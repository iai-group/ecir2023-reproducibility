"""ANCE dense retrieval."""

import os
import shutil

import pyterrier as pt
import pyterrier_ance
from treccast.core.base import Query, ScoredDocument
from treccast.core.ranking import Ranking
from treccast.retriever.retriever import Retriever

_DENSE_RETRIEVAL_MODEL_CHECKPOINT = (
    "data/retrieval/ance/Passage ANCE(FirstP) Checkpoint"
)
_DEFAULT_LOCATION_OF_ANCE_INDEX = "data/retrieval/ance/anceindex"


class ANCEDenseRetriever(Retriever):
    def __init__(
        self, collection_id: str, num_docs: int, reset_index: bool = True
    ) -> None:
        """Initializes ANCE dense retrieval model.

        We use the following default model checkpoint:
        https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/
          OpenSource/Passage_ANCE_FirstP_Checkpoint.zip

        Args:
            collection_id: Id of the ir-datasets collection.
            num_docs: Number of documents to be included in the index.
            reset_index: If true, resets the index that is in the default
              location. Defaults to True.
        """
        self._dataset = pt.get_dataset(collection_id)
        self._num_docs = num_docs

        if reset_index:
            shutil.rmtree(_DEFAULT_LOCATION_OF_ANCE_INDEX)

        if not os.path.isdir(_DEFAULT_LOCATION_OF_ANCE_INDEX):
            indexer = pyterrier_ance.ANCEIndexer(
                checkpoint_path=_DENSE_RETRIEVAL_MODEL_CHECKPOINT,
                index_path=_DEFAULT_LOCATION_OF_ANCE_INDEX,
                num_docs=self._num_docs,
            )
            indexer.index(self._dataset.get_corpus_iter())
            del indexer

        self.ance_retriever = pyterrier_ance.ANCERetrieval(
            checkpoint_path=_DENSE_RETRIEVAL_MODEL_CHECKPOINT,
            index_path=_DEFAULT_LOCATION_OF_ANCE_INDEX,
        )

    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Performs retrieval.

        Args:
            query: Query instance.
            num_results: Number of documents to return (defaults
              to 1000).

        Returns:
            Document ranking.
        """

        print("Retrieving using query: ", query.question)

        retrieved_results = (self.ance_retriever % num_results).search(
            query.question
        )
        # Adding text of the retrieved passages to the results
        pipe = retrieved_results >> pt.text.get_text(self._dataset, ["text"])
        topics = self._dataset.get_topics()
        retrieved_results = pipe.transform(topics)
        retrieved_results = retrieved_results.sort_values(
            "rank", ascending=True
        )

        return Ranking(
            query.query_id,
            [
                ScoredDocument(
                    retrieved_results["docid"][id],
                    retrieved_results["text"][id],
                    retrieved_results["score"][id],
                )
                for id in retrieved_results.index
            ],
        )


if __name__ == "__main__":
    # Example usage.

    # Before running the script:
    # - Make sure you have Java installed
    # - Update conda environment by running:
    # conda env update --file environment.yaml --prune
    # - Run the following command:
    # pip install --upgrade \
    #   git+https://github.com/terrierteam/pyterrier_ance.git

    if not pt.started():
        pt.init()

    # ance = ANCEDenseRetriever("irds:msmarco-passage", 8841823)
    ance = ANCEDenseRetriever(
        collection_id="irds:vaswani", num_docs=11429, reset_index=True
    )

    query = Query("1", "What is a chemical reaction?")
    ranking = ance.retrieve(query, 10)
    for rank, doc in enumerate(ranking.fetch_topk_docs(10), start=1):
        print(f"{rank}: {doc.score}, {doc.doc_id}, {doc.content}")
