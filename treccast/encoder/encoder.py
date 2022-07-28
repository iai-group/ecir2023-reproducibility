"""Interface for text encoding."""

import logging
import operator
import os
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from treccast.core.util.data_generator import DataGeneratorMixin

# Batch of data (id, text) to encode.
_BatchIterator = Iterator[List[Tuple[str, str]]]
# ACTION is used to select the correct format to generate data to be encoded.
ACTION = "encoding"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)


class Encoder(DataGeneratorMixin, ABC):
    def __init__(
        self, batch_size: int = 50, embedding_dim: Optional[int] = 100
    ) -> None:
        """Abstract class for embedding generation.

        Args:
            batch_size (optional): Size of the batch to encode. Defaults to 50.
            embedding_dim (optional): Number of dimensions for embedding vector.
              Defaults to 100.
        """
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Interface for text encoding that needs to be implemented.

        Args:
            texts: Texts to encode.

        Returns:
            Texts embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self, filepath: str, passage_ids: List[str], embeddings: np.ndarray
    ) -> None:
        """Interface for saving passage ids and embeddings to file.

        Args:
            filepath: Location to saving file.
            passage_ids: List of passage id.
            embeddings: Embeddings for passage ids.
        """
        raise NotImplementedError

    def generate_embeddings(
        self, batches: _BatchIterator, filepath: str
    ) -> None:
        """Generates and saves embeddings for batches of texts.

        Args:
            batches: Batches iterator.
            filepath: Location of output file.
        """
        for batch in batches:
            passage_ids = list(map(operator.itemgetter(0), batch))
            passages_texts = list(map(operator.itemgetter(1), batch))
            embeddings = self.encode(passages_texts)
            self.save(filepath, passage_ids, embeddings)

        logging.info("Embeddings saved at: %s", filepath)


class TransformersEncoder(Encoder, ABC):
    def __init__(
        self,
        batch_size: int = 50,
        max_length: int = 512,
        padding: str = "longest",
        truncation: bool = True,
        add_special_tokens: bool = True,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """Transformers encoder.

        Args:
            batch_size (optional): Size of the batch to encode. Defaults to 50.
            max_length (optional): Maximal number of tokens. Defaults to 512
              tokens.
            padding (optional): Padding strategy. Defaults to "longest".
            truncation (optional): Allow truncation. Defaults to True.
            add_special_tokens (optional): Add a dictionary of special tokens.
              Defaults to True.
            embedding_dim (optional): Number of dimensions for embedding vector.
              Defaults to None.
        """
        super().__init__(batch_size, embedding_dim)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._max_length = max_length
        self._padding = padding
        self._truncation = truncation
        self._add_special_tokens = add_special_tokens

    def save(
        self, filepath: str, passage_ids: List[str], embeddings: np.ndarray
    ) -> None:
        """Saves passage ids and embeddings to file.

        Args:
            filepath: Location to saving file.
            passage_ids: List of passage id.
            embeddings: Embeddings for passage ids.

        Raises:
            ValueError: File type not supported. Supported type: .hdf5
        """

        if not filepath.endswith(".hdf5"):
            raise ValueError("File type not supported. Supported type: .hdf5")

        # Convert str ids to bytes
        passage_ids = np.array(passage_ids, dtype="|S3")

        if not os.path.exists(filepath):
            file = h5py.File(filepath, "w")
            file.create_dataset(
                "embeddings",
                data=embeddings,
                compression="gzip",
                chunks=True,
                maxshape=(None, embeddings.shape[1]),
            )
            file.create_dataset(
                "passage_ids",
                data=passage_ids,
                compression="gzip",
                chunks=True,
                maxshape=(None,),
            )
        else:
            file = h5py.File(filepath, "a")
            file["embeddings"].resize(
                (file["embeddings"].shape[0] + embeddings.shape[0]), axis=0
            )
            file["embeddings"][-embeddings.shape[0] :] = embeddings

            file["passage_ids"].resize(
                (file["passage_ids"].shape[0] + passage_ids.shape[0]), axis=0
            )
            file["passage_ids"][-passage_ids.shape[0] :] = passage_ids

        file.close()
