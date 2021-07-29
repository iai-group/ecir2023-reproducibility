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
from typing import Dict, Iterator, Union

from trec_car import read_data
from elasticsearch.helpers import parallel_bulk
from treccast.core.collection import ElasticSearchIndex

DEFAULT_MS_MARCO_PASSAGE_DATASET = (
    "/data/collections/msmarco-passage/collection.tsv"
)
DEFAULT_TREC_CAR_PARAGRAPH_DATASET = (
    "/data/collections/trec-car/paragraphCorpus/dedup.articles-paragraphs.cbor"
)
DEFAULT_INDEX_NAME = "ms_marco_trec_car"
DEFAULT_HOST_NAME = "localhost:9204"


class Indexer(ElasticSearchIndex):
    def __init__(
        self, index_name: str, hostname: str = "localhost:9200"
    ) -> None:
        """Initializes an Elasticsearch instance on a given host.

        Args:
            index_name: Index name.
            hostname: Host name and port (defaults to
                "localhost:9200").
        """
        super().__init__(index_name, hostname)

    def generate_data_marco(
        self, filepath: str
    ) -> Iterator[Dict[str, Union[str, Dict]]]:
        """Data generator for batch indexing of MS MARCO dataset.

        Args:
            filepath: Path to the MS MARCO passage dataset

        Yields:
            Iterator[dict]: Dictionary containing index, id and contents of a
                passage.
        """
        print("Starting to index the MS MARCO passage dataset")
        with open(filepath) as f:
            for i, line in enumerate(f):
                pid, content = line.strip().split("\t")
                yield {
                    "_index": self._index_name,
                    "_id": f"MARCO_{pid}",
                    "_source": {"body": content},
                }
                if i % 1000000 == 0:
                    print(f"Indexed {i} paragraphs")
            print(f"Indexing finished. Indexed total {i} paragraphs.\n")

    def generate_data_car(
        self, filepath: str
    ) -> Iterator[Dict[str, Union[str, Dict]]]:
        """Data generator for batch indexing of TREC CAR dataset.

        Args:
            filepath: Path to the TREC CAR paragraph dataset

        Yields:
            Iterator[dict]: Dictionary containing index, id and contents of a
                paragraph.
        """
        print("Starting to index the TREC CAR paragraphs dataset")
        with open(filepath, "rb") as f:
            for i, paragraph in enumerate(read_data.iter_paragraphs(f)):
                # Paragraphs contain text with named entities in format
                # [Text](Named Entity) under paragraph.bodies.
                yield {
                    "_index": self._index_name,
                    "_id": f"CAR_{paragraph.para_id}",
                    "_source": {"body": paragraph.get_text().strip()},
                }
                if i % 1000000 == 0:
                    print(f"Indexed {i} paragraphs")
            print(f"Indexing finished. Indexed total {i} paragraphs.")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    indexing = Indexer(args.index, args.host)
    if args.reset:
        indexing.reset_index()
    if args.ms_marco:
        data_generator = indexing.generate_data_marco(args.ms_marco)
        indexing.batch_index(data_generator)
    if args.trec_car:
        data_generator = indexing.generate_data_car(args.trec_car)
        indexing.batch_index(data_generator)
