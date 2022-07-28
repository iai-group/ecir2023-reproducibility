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
from typing import Any, Dict, Iterator

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


class Indexer(DataGeneratorMixin, ElasticSearchIndex):
    def __init__(
        self,
        index_name: str,
        hostname: str = "localhost:9200",
        use_docT5query: bool = False,
        docT5query_n_queries: int = 3,
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name: Index name.
            hostname: Host name and port (defaults to
              "localhost:9200"),
            use_docT5query: Use DocT5Query to expand passages or not.
            docT5query_n_queries: Num queries to predict per passage.
        """
        super().__init__(
            index_name,
            hostname,
            timeout=120,
            max_retries=10,
            retry_on_timeout=True,
        )

        self.docT5query = (
            DocT5Query(docT5query_n_queries) if use_docT5query else None
        )

    def process_documents(
        self, data_generator: Iterator[dict]
    ) -> Iterator[dict]:
        """Adds elasticsearch specific information to generated documents.

        Args:
            data_generator: Document generator.

        Yields:
            Processed documents.
        """
        for document in data_generator:
            if self.docT5query:
                doc2query_queries = self.docT5query.predict_queries(
                    document["_source"]["body"]
                )
                document["_source"]["doc2query"] = " ".join(doc2query_queries)
            yield document

    def batch_index(self, data_generator: Iterator[dict]) -> None:
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
        "--docT5query",
        action="store_true",
        help="Expands passages with docT5query",
    )
    parser.add_argument(
        "--docT5query_n_queries",
        type=int,
        default=3,
        help="How many queries to predict for each passage",
    )
    return parser.parse_args()


def main(args):
    """Index documents based on the command line arguments.

    Args:
        args: Arguments.
    """
    indexing = Indexer(
        args.index, args.host, args.docT5query, args.docT5query_n_queries
    )
    if args.reset:
        indexing.delete_index()

    indexing.create_index(use_analyzer=not args.no_analyzer)

    data_generators = []
    if args.ms_marco:
        data_generators.append(
            indexing.generate_data_marco(
                _ACTION, args.ms_marco, index_name=indexing._index_name
            )
        )

    if args.trec_car:
        data_generators.append(
            indexing.generate_data_car(
                _ACTION, args.trec_car, index_name=indexing._index_name
            )
        )
    if args.trecweb:
        for filepath in args.trecweb:
            data_generators.append(
                indexing.generate_data_trecweb(
                    _ACTION, filepath, index_name=indexing._index_name
                )
            )

    data_generator = itertools.chain(*data_generators)
    indexing.batch_index(indexing.process_documents(data_generator))


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
