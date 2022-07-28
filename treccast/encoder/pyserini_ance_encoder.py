"""Text encoding with Pyserini ANCE encoder model."""

import argparse
import os
from typing import List, Optional, Union

import numpy as np
import torch
from pyserini.encode import AnceEncoder
from transformers import AutoTokenizer
from treccast.encoder.encoder import ACTION, TransformersEncoder

_DEFAULT_OUTPUT_FILE = "data/embeddings/trec-cast-embeddings.hdf5"
_DEFAULT_MS_MARCO_PASSAGE_DATASET = (
    "/data/collections/msmarco-passage/collection.tar.gz"
)
_DEFAULT_TREC_CAR_PARAGRAPH_DATASET = (
    "/data/collections/trec-car/paragraphCorpus/dedup.articles-paragraphs.cbor"
)
_DEFAULT_ENCODER_PATH = "castorini/ance-msmarco-passage"
_DEFAULT_BATCH_SIZE = 500
_DEFAULT_MAX_LENGTH = 512
_DEFAULT_PADDING = "longest"


class PyseriniAnceEncoder(TransformersEncoder):
    def __init__(
        self,
        model_path: str = "castorini/ance-msmarco-passage",
        tokenizer_path: Optional[str] = None,
        batch_size: int = 50,
        max_length: int = 512,
        padding: str = "longest",
        truncation: bool = True,
        add_special_tokens: bool = True,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """Pyserini ANCE encoder.

        Args:
            model_path (optional): Hugging Face model name. Defaults to
              "castorini/ance-msmarco-passage".
            tokenizer_path (optional): Hugging Face tokenizer model name.
              Defaults to None.
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
        super().__init__(
            batch_size,
            max_length,
            padding,
            truncation,
            add_special_tokens,
            embedding_dim,
        )

        tokenizer_path = tokenizer_path or model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self._model = AnceEncoder.from_pretrained(model_path).to(
            self._device, non_blocking=True
        )

        self._embedding_dim = self._model.embeddingHead.out_features

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Performs encoding of text.

        Args:
            texts: Texts to encode.

        Returns:
            Texts embeddings.
        """
        texts = [texts] if isinstance(texts, str) else texts

        inputs = self._tokenizer(
            texts,
            max_length=self._max_length,
            padding=self._padding,
            truncation=self._truncation,
            add_special_tokens=self._add_special_tokens,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(inputs["input_ids"]).to(
            self._device, non_blocking=True
        )
        attention_mask = torch.tensor(inputs["attention_mask"]).to(
            self._device, non_blocking=True
        )
        embeddings = (
            self._model(input_ids, attention_mask).detach().cpu().numpy()
        )

        return embeddings


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="pyserini_encoder.py")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=_DEFAULT_OUTPUT_FILE,
        help="Specifies the path to the output file",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Removes output file before starting",
    )
    # Args related to data files
    parser.add_argument(
        "-m",
        "--ms_marco",
        type=str,
        nargs="?",
        const=_DEFAULT_MS_MARCO_PASSAGE_DATASET,
        help="Specifies the path to MS MARCO dataset",
    )
    parser.add_argument(
        "--trec_car",
        type=str,
        nargs="?",
        const=_DEFAULT_TREC_CAR_PARAGRAPH_DATASET,
        help="Specifies the path to TREC CAR dataset",
    )
    parser.add_argument(
        "--trecweb",
        type=str,
        nargs="+",
        help="Specifies the path(s) to TRECWEB dataset(s)",
    )
    # Args related to TransformersEncoder
    parser.add_argument(
        "--encoder",
        type=str,
        default=_DEFAULT_ENCODER_PATH,
        help="Specifies the path to the encoder model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        nargs="?",
        help="Specifies the path to the tokenizer model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Specifies the batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=_DEFAULT_MAX_LENGTH,
        help="Specifies the maximum number of tokens",
    )
    parser.add_argument(
        "--pad",
        choices=["longest", "max_length", "do_not_pad"],
        default=_DEFAULT_PADDING,
        help="Specifies the padding strategy",
    )
    parser.add_argument(
        "--trunc",
        action="store_false",
        help="Specifies whether truncation is applied or not",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_false",
        help="Specifies whether special tokens are added or not",
    )
    parser.add_argument(
        "--embedding-dim",
        nargs="?",
        type=int,
        help="Specifies the number of dimensions for embeddings vector",
    )
    return parser.parse_args()


def main(args):
    """Encodes passages based on the commad line arguments.

    Args:
        args: Arguments.
    """
    encoder = PyseriniAnceEncoder(
        args.encoder,
        args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        padding=args.pad,
        truncation=args.trunc,
        add_special_tokens=args.add_special_tokens,
        embedding_dim=args.embedding_dim,
    )
    if args.clean and os.path.exists(args.output_file):
        os.remove(args.output_file)

    if args.ms_marco:
        data_generator = encoder.generate_data_marco(ACTION, args.ms_marco)
        data_batches = encoder.generate_batches(
            data_generator, encoder._batch_size
        )
        encoder.generate_embeddings(data_batches, args.output_file)

    if args.trec_car:
        data_generator = encoder.generate_data_car(ACTION, args.trec_car)
        data_batches = encoder.generate_batches(
            data_generator, encoder._batch_size
        )
        encoder.generate_embeddings(data_batches, args.output_file)

    if args.trecweb:
        for filepath in args.trecweb:
            data_generator = encoder.generate_data_trecweb(ACTION, filepath)
            data_batches = encoder.generate_batches(
                data_generator, encoder._batch_size
            )
            encoder.generate_embeddings(data_batches, args.output_file)


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
