"""ANCE dense retrieval."""

import logging
import os
import shutil
from itertools import chain

import pandas as pd
import pyterrier as pt
import pyterrier_ance
from treccast.core.base import Query, ScoredDocument
from treccast.core.ranking import Ranking
from treccast.core.util.file_parser import FileParser
from treccast.core.util.passage_loader import PassageLoader
from treccast.retriever.retriever import Retriever

_DENSE_RETRIEVAL_MODEL_CHECKPOINT = (
    "data/retrieval/ance/Passage ANCE(FirstP) Checkpoint"
)
_DEFAULT_LOCATION_OF_ANCE_INDEX = (
    "data/retrieval/ance/trecweb_ms_marco_kilt_wapo_ance"
)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)


class ANCEDenseRetriever(Retriever):
    def __init__(
        self,
        index_path: str = _DEFAULT_LOCATION_OF_ANCE_INDEX,
        year: str = "2021",
        reset_index: bool = False,
        es_host_name: str = "localhost:9204",
        es_index_name: str = "ms_marco_kilt_wapo_clean",
        k: int = 1000,
        collections: str = "/data/collections/trec-cast",
    ) -> None:
        """Initializes ANCE dense retrieval model.

        We use the following default model checkpoint:
        https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip # noqa

        Args:
            index_path: Path to the ANCE index. Defaults to
              "data/retrieval/ance/ms_marco_kilt_wapo_ance".
            year: Year for which to run the retrieval. Defaults to 2021.
            reset_index (optional): If true, resets the index that is in the
              default location. Defaults to False.
            es_host_name (optional): Name of host and port number for passage
              loader. Defaults to "localhost:9204".
            es_index_name (optional): Name of index for passage loader. Defaults
              to "ms_marco_kilt_wapo_clean".
            k (optional): Number of documents to retrieve for each query.
              Defaults to 1000.
            collections: Path to the directory containing trecweb files.
        """
        if year == "2021":
            self._kilt_dataset = pt.get_dataset("irds:kilt")
            self._marco_document_dataset = pt.get_dataset(
                "irds:msmarco-document"
            )
            self._dataset_iter = chain(
                self.trecweb_file_generator(
                    f"{collections}/kilt_knowledgesource.trecweb"
                ),
                self.trecweb_file_generator(
                    f"{collections}/msmarco-docs.trecweb"
                ),
                self.trecweb_file_generator(
                    f"{collections}/TREC_Washington_Post_collection.v4.trecweb"
                ),
            )
        elif year == "2020":
            self._trec_car_dataset = pt.get_dataset("irds:car/v2.0")
            self._marco_passage_dataset = pt.get_dataset("irds:msmarco-passage")
            self._dataset_iter = chain(
                self.trec_car_generator(),
                self.ms_marco_passage_generator(),
            )

        if reset_index and os.path.isdir(index_path):
            logging.info("--- Resetting index ---")
            shutil.rmtree(index_path)

        if not os.path.isdir(index_path):
            logging.info("--- Starting indexing ---")
            indexer = pyterrier_ance.ANCEIndexer(
                checkpoint_path=_DENSE_RETRIEVAL_MODEL_CHECKPOINT,
                index_path=index_path,
                verbose=False,
            )
            indexer.index(self._dataset_iter)
            del indexer

        self.ance_retriever = pyterrier_ance.ANCERetrieval(
            checkpoint_path=_DENSE_RETRIEVAL_MODEL_CHECKPOINT,
            index_path=index_path,
            num_results=k,
        )

        self._passage_loader = PassageLoader(es_host_name, es_index_name)

    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Performs retrieval.

        Args:
            query: Query instance.
            num_results (optional): Number of documents to return (defaults
              to 1000).

        Returns:
            Document ranking.
        """

        print(
            "Retrieving using query:\n",
            str(query),
            "\n",
        )

        retrieved_results = (self.ance_retriever % num_results).search(
            query.question
        )
        retrieved_results = retrieved_results.reset_index()

        ranking = Ranking(query.query_id)
        for _, row in retrieved_results.iterrows():
            content = self._passage_loader.get(row["docno"])
            if content is not None:
                ranking.add_doc(
                    ScoredDocument(
                        doc_id=row["docno"],
                        score=row["score"],
                        doc_content=content,
                    )
                )
        return ranking

    def trecweb_file_generator(self, filepath: str):
        for _, (passage_id, _, passage) in enumerate(
            FileParser.parse(filepath)
        ):
            yield {"docno": passage_id, "text": passage}

    def trec_car_generator(self):
        for doc in self._trec_car_dataset.get_corpus_iter(verbose=False):
            docno = "CAR_" + str(doc["docno"])
            text = doc["text"].replace("\n", " ")

            yield {"docno": docno, "text": text}

    def ms_marco_passage_generator(self):
        for doc in self._marco_passage_dataset.get_corpus_iter(verbose=False):
            docno = "MARCO_" + str(doc["docno"])
            text = doc["text"].replace("\n", " ")

            yield {"docno": docno, "text": text}


if __name__ == "__main__":
    # Example usage.
    #
    # Before running the script:
    # - Make sure you have Java installed
    # - Update conda environment by running:
    # conda env update --file environment.yaml --prune
    # - Run the following command:
    # pip install --upgrade git+https://github.com/WerLaj/pyterrier_ance.git

    if not pt.started():
        pt.init()

    ance = ANCEDenseRetriever()

    query = Query(
        "81_1", "How do you know when your garage door opener is going bad?"
    )
    topic = pd.DataFrame({"qid": "1", "query": query.question}, index=[0])
    ranking = ance.retrieve(query, 1000)
    for rank, doc in enumerate(ranking.fetch_topk_docs(50), start=1):
        print(f"{rank}: {doc.score}, {doc.doc_id}, {doc.content}")
