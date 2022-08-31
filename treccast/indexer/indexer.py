"""Implements abstract class with interfaces of bulk indexing data collections.

Usage:
    Reseting the default index and indexing MS MARCO and TREC CAST datasets
    using default paths.

    $ python indexer.py --ms_marco --trec_car --reset

    Indexing only MS MARCO with a custom path to the dataset without resetting
    the index.

    $ python indexer.py --ms_marco path/to/collection

    Indexing only MS MARCO with a custom path to the dataset without resetting
    the index, with expanding the passages with docT5query predictions:

    $ python indexer.py --ms_marco path/to/collection --docT5query
"""
import argparse
import itertools
from typing import Any, Dict, Iterator, List

import nltk
from elasticsearch.helpers import parallel_bulk
from nltk.corpus import stopwords
from treccast.core.collection import ElasticSearchIndex
from treccast.core.util.data_generator import DataGeneratorMixin
from treccast.indexer.docT5query import DocT5Query

DEFAULT_MS_MARCO_PASSAGE_DATASET = (
    "/data/collections/msmarco-passage/collection.tar.gz"
)
DEFAULT_TREC_CAR_PARAGRAPH_DATASET = (
    "/data/collections/trec-car/paragraphCorpus/dedup.articles-paragraphs.cbor"
)
DEFAULT_INDEX_NAME = "ms_marco_trec_car"
DEFAULT_HOST_NAME = "localhost:9204"
_ACTION = "indexing"
_BATCH_SIZE = 30

_DataIterator = Iterator[dict]
_BatchIterator = Iterator[List[dict]]


class Indexer(DataGeneratorMixin, ElasticSearchIndex):
    def __init__(
        self,
        index_name: str,
        hostname: str = "localhost:9200",
        docT5query_n_queries: int = 0,
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name: Index name.
            hostname: Host name and port (defaults to
              "localhost:9200"),
            docT5query_n_queries (optional): Num queries to predict per passage.
              If 0, do not use passage expansion. Defaults to 0.
        """
        super().__init__(
            index_name,
            hostname,
            timeout=120,
            max_retries=10,
            retry_on_timeout=True,
        )

        self.docT5query = (
            DocT5Query(docT5query_n_queries) if docT5query_n_queries else None
        )

    def _process_documents(
        self, data_generator: _DataIterator
    ) -> _DataIterator:
        """Adds elasticsearch specific information to generated documents.

        Args:
            data_generator: Document generator.

        Yields:
            Processed documents.
        """
        for document in data_generator:
            document["_index"] = self.index_name
            yield document

    def _process_document_batches(
        self, batches: _BatchIterator
    ) -> _DataIterator:
        """Expands documents with doc2query.

        Args:
            batches: Document batch generator.

        Yields:
            Processed (expanded) documents.
        """
        for batch in batches:
            doc2query_queries = self.docT5query.batch_generate_queries(
                [
                    document.get("catch_all", document["body"])
                    for document in batch
                ]
            )
            for document, document_queries in zip(batch, doc2query_queries):
                document["_index"] = self.index_name
                document["doc2query"] = " ".join(document_queries)
                document["catch_all"] = (
                    document.get("catch_all", document["body"])
                    + " "
                    + document["doc2query"]
                )
                yield document

    def process_documents(
        self, data_generator: _DataIterator, batch_size: int = _BATCH_SIZE
    ) -> _DataIterator:
        """Adds elasticsearch specific information to generated documents.

        If docT5query is used, additionally expands documents with queries.

        Args:
            data_generator: Document generator.
            batch_size (optional): Batch size. Used for more efficient dot2query
              expansions. Defaults to 30

        Returns:
            Data iterator that processes documents.
        """
        if self.docT5query:
            batches = self.generate_batches(data_generator, batch_size)
            return self._process_document_batches(batches)
        return self._process_documents(data_generator)

    def batch_index(self, data_generator: _DataIterator) -> None:
        """Bulk index a dataset in parallel.

        Args:
            data_generator: Data generator. Should yield index
              name, id and contents to index.
        """
        for success, info in parallel_bulk(
            self._es,
            data_generator,
            thread_count=12,
            chunk_size=5000,
            max_chunk_bytes=104857600,
            queue_size=6,
        ):
            if not success:
                print("A document failed:", info)

    def _get_analysis_settings(self) -> Dict[str, Any]:
        """Elasticsearch analyzer with lowercase tokenization, stopword removal
        and KStemming.

        Returns:
            Dictionary containing Elasticsearch configuration.
        """
        # Stop words need to be downloaded if they are not already.
        nltk.download("stopwords")

        # TODO make tests for index using this analyzer
        # https://github.com/iai-group/trec-cast-2021/issues/70
        return {
            "analysis": {
                "analyzer": {
                    "default": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "nltk_stop", "kstem"],
                    },
                },
                "filter": {
                    "nltk_stop": {
                        "type": "stop",
                        "stopwords": stopwords.words("english"),
                    }
                },
            }
        }


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="indexing.py")
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Specifies the index name",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST_NAME,
        help="Specifies the hostname and the port",
    )
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Reset index",
    )
    parser.add_argument(
        "--no-analyzer",
        action="store_true",
        help="Do not use custom Elasticsearch analyzer",
    )
    parser.add_argument(
        "-m",
        "--ms_marco",
        type=str,
        nargs="?",
        const=DEFAULT_MS_MARCO_PASSAGE_DATASET,
        help="Specifies the path to MS MARCO dataset",
    )
    parser.add_argument(
        "-c",
        "--trec_car",
        type=str,
        nargs="?",
        const=DEFAULT_TREC_CAR_PARAGRAPH_DATASET,
        help="Specifies the path to TREC CAR dataset",
    )
    parser.add_argument(
        "--trecweb",
        type=str,
        nargs="+",
        help="Specifies the path(s) to TRECWEB dataset(s)",
    )
    parser.add_argument(
        "--docT5query_n_queries",
        type=int,
        default=0,
        help=(
            "How many queries to predict for each passage using docT5query."
            "Defaults to 0 (Means no expansion)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_BATCH_SIZE,
        help="Batch size to use for doc2query.",
    )
    return parser.parse_args()


def main(args):
    """Index documents based on the command line arguments.

    Args:
        args: Arguments.
    """
    indexing = Indexer(args.index, args.host, args.docT5query_n_queries)
    if args.reset:
        indexing.delete_index()

    indexing.create_index(use_analyzer=not args.no_analyzer)

    data_generators = []
    if args.ms_marco:
        data_generators.append(
            indexing.generate_data_marco(_ACTION, args.ms_marco)
        )

    if args.trec_car:
        data_generators.append(
            indexing.generate_data_car(_ACTION, args.trec_car)
        )
    if args.trecweb:
        for filepath in args.trecweb:
            data_generators.append(
                indexing.generate_data_trecweb(_ACTION, filepath)
            )

    data_generator = itertools.chain(*data_generators)
    documents = indexing.process_documents(data_generator)
    indexing.batch_index(documents)


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
