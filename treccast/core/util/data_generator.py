"""Mixin class to generate data from file.

Action supported: encoding and indexing.
"""

import itertools
import logging
from typing import Any, Iterator, List, Optional

from trec_car import read_data
from treccast.core.util.file_parser import FileParser

_DataIterator = Iterator[Any]
_BatchIterator = Iterator[List[Any]]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)


class DataGeneratorMixin:
    def generate_data_marco(
        self, action: str, filepath: str, index_name: Optional[str] = None
    ) -> _DataIterator:
        """Data generator for batch encoding of MS MARCO dataset.

        Args:
            action: Action executed after data generation.
            filepath: Path to the MS MARCO passage dataset.
            index_name (optional): Elasticsearch index name. Defaults to None.

        Yields:
            Iterator[Any]: Object containing id and contents of a passage.

        Raises:
            RuntimeError: supported actions are encoding and indexing.
        """
        logging.info("Starting to generate data for MS MARCO passage dataset.")
        for i, line in enumerate(FileParser.parse(filepath)):
            pid, content = line.split("\t")
            if action == "encoding":
                yield (f"MARCO_{pid}", content)
            elif action == "indexing":
                yield {
                    "_index": index_name,
                    "_id": f"MARCO_{pid}",
                    "_source": {"body": content},
                }
            else:
                raise RuntimeError(
                    "Cannot generate data. Supported actions: indexing and "
                    "encoding."
                )
            if i % 1000000 == 0:
                logging.info("Generated %s paragraphs", i)
        logging.info("Generation finished. Generated total %s paragraphs.", i)

    def generate_data_car(
        self, action: str, filepath: str, index_name: Optional[str] = None
    ) -> _DataIterator:
        """Data generator for batch encoding of TREC CAR dataset.

        Args:
            action: Action executed after data generation.
            filepath: Path to the TREC CAR paragraph dataset.
            index_name (optional): Elasticsearch index name. Defaults to None.

        Yields:
            Iterator[Any]: Object containing id and contents of a passage.

        Raises:
            RuntimeError: supported actions are encoding and indexing.
        """
        logging.info(
            "Starting to generate data for TREC CAR paragraphs dataset."
        )
        with open(filepath, "rb") as f:
            for i, paragraph in enumerate(read_data.iter_paragraphs(f)):
                # Paragraphs contain text with named entities in format
                # [Text](Named Entity) under paragraph.bodies.
                if action == "encoding":
                    yield (
                        f"CAR_{paragraph.para_id}",
                        paragraph.get_text().strip(),
                    )
                elif action == "indexing":
                    yield {
                        "_index": index_name,
                        "_id": f"CAR_{paragraph.para_id}",
                        "_source": {"body": paragraph.get_text().strip()},
                    }
                else:
                    raise RuntimeError(
                        "Cannot generate data. Supported actions: indexing and "
                        "encoding."
                    )
                if i % 1000000 == 0:
                    logging.info("Generated %s paragraphs", i)
            logging.info(
                "Generation finished. Generated total %s paragraphs.", i
            )

    def generate_data_trecweb(
        self, action: str, filepath: str, index_name: Optional[str] = None
    ) -> _DataIterator:
        """Data generator for batch encoding of preprocessed TRECWEB files.

        Args:
            action: Action executed after data generation.
            filepath: Path to a TRECWEB dataset.
            index_name (optional): Elasticsearch index name. Defaults to None.

        Yields:
            Iterator[Any]: Object containing id and contents of a passage.

        Raises:
            RuntimeError: supported actions are encoding and indexing.
        """
        logging.info("Starting to generate data for filepath: %s", filepath)
        for i, (passage_id, title, passage) in enumerate(
            FileParser.parse(filepath)
        ):
            if action == "encoding":
                yield (passage_id, f"{title} {passage}")
            elif action == "indexing":
                yield {
                    "_index": index_name,
                    "_id": passage_id,
                    "_source": {
                        "body": passage,
                        "title": title,
                        "catch_all": f"{title} {passage}",
                    },
                }
            else:
                raise RuntimeError(
                    "Cannot generate data. Supported actions: indexing and "
                    "encoding."
                )

            if i % 1000000 == 0:
                logging.info("Generated %s paragraphs", i)
        logging.info("Generation finished. Generated total %s paragraphs.", i)

    def generate_batches(
        self, iterator: _DataIterator, batch_size: int
    ) -> _BatchIterator:
        """Splits data into batches.

        Args:
            iterator: Iterator containing passages objects.
            batch_size: Size of a batch.

        Yields:
            Iterator[List]: List of passages objects.
        """
        data_remaining = True
        while data_remaining:
            chunk = list(itertools.islice(iterator, batch_size))
            if not chunk:
                data_remaining = False
                continue
            yield chunk
