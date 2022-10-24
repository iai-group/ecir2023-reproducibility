"""Implements abstract class with interfaces of bulk indexing data collections.

Usage:
    Reseting the default index and indexing MS MARCO and TREC CAST datasets
    using default paths.

    $ python indexer.py --ms_marco --trec_car --reset

    Indexing only MS MARCO with a custom path to the dataset without resetting
    the index.

    $ python indexer.py --ms_marco path/to/collection
"""
import argparse
import itertools
from typing import Any, Dict, Iterator

import nltk
from elasticsearch.helpers import parallel_bulk
from nltk.corpus import stopwords
from treccast.core.collection import ElasticSearchIndex
from treccast.core.util.data_generator import DataGeneratorMixin

DEFAULT_MS_MARCO_PASSAGE_DATASET = (
    "/data/collections/collection.tar.gz"
)
DEFAULT_TREC_CAR_PARAGRAPH_DATASET = (
    "/data/collections/dedup.articles-paragraphs.cbor"
)
DEFAULT_INDEX_NAME = "ms_marco_trec_car_clean"
DEFAULT_HOST_NAME = "localhost:9204"
_ACTION = "indexing"

_DataIterator = Iterator[dict]


class Indexer(DataGeneratorMixin, ElasticSearchIndex):
    def __init__(
        self, index_name: str, hostname: str = "localhost:9204"
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name: Index name.
            hostname: Host name and port (defaults to
              "localhost:9204"),
        """
        super().__init__(
            index_name,
            hostname,
            timeout=120,
            max_retries=10,
            retry_on_timeout=True,
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

    def process_documents(self, data_generator: _DataIterator) -> _DataIterator:
        """Adds elasticsearch specific information to generated documents.

        Args:
            data_generator: Document generator.

        Returns:
            Data iterator that processes documents.
        """
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
    return parser.parse_args()


def main(args):
    """Index documents based on the command line arguments.

    Args:
        args: Arguments.
    """
    indexing = Indexer(args.index, args.host)
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
