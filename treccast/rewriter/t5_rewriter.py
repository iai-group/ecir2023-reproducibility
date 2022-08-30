"""Query rewriter based on historical context using T5."""

import argparse
import csv
import json
from typing import Dict, Union

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from treccast.core import NEURAL_MODEL_CACHE_DIR
from treccast.core.base import Context, Query, SparseQuery
from treccast.core.topic import QueryRewrite, Topic
from treccast.core.util.passage_loader import PassageLoader
from treccast.rewriter.rewriter import Rewriter

# The path to the fine-tuned model to be loaded and used for rewriting.
_MODEL_DIR = (
    "data/fine_tuning/rewriter/qrecc/T5_QReCC_st_WaterlooClarke-train/"
    + "best_model"
)
# Year for which the rewrites should be generated.
_YEAR = "2021"
# Max sequence length for the model.
_MAX_LENGTH = 512
# The path to the output directory for the generated query rewrites.
_OUTPUT_DIR = "data/rewrites/2021/12_T5_QReCC.tsv"
# Specifies whether previously rewritten queries should be used in the context
# for rewriting current query.
_USE_PREVIOUS_REWRITTEN_UTTERANCE = True
# Specifies whether the last canonical response should be used in the context
# for rewriting current query.
_USE_CANONICAL_RESPONSE = True
# The name of the index to be used for loading canonical responses.
_INDEX_NAME = "ms_marco_kilt_wapo_clean"
# Special token to separate input sequences.
_SEPARATOR = "<sep>"
# Number of beams to use in beam search.
_NUM_BEAMS = 10


class T5Rewriter(Rewriter):
    def __init__(
        self,
        model_name: str = "castorini/t5-base-canard",
        max_length: int = 64,
        num_beams: int = _NUM_BEAMS,
        early_stopping: bool = True,
        sparse: bool = False,
    ) -> None:
        """Instantiate T5 rewriter.

        Args:
            model_name (optional): Hugging Face model name. Defaults to
              "castorini/t5-base-canard".
            max_length (optional): Max sequence length. Default model was
              trained with sequences up to 64. Defaults to 64.
            num_beams (optional): How many beams to explore. Defaults to 10.
            early_stopping (optional): Whether to stop when num_beams is
              reached. Defaults to True.
            sparse (optional): If true performs sparse query rewrite. Defaults
              to False.
        """
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._max_length = max_length
        self._num_beams = num_beams
        self._early_stopping = early_stopping
        self.sparse = sparse

        self._tokenizer = T5Tokenizer.from_pretrained(
            model_name, cache_dir=NEURAL_MODEL_CACHE_DIR
        )
        self._model = (
            T5ForConditionalGeneration.from_pretrained(
                model_name, cache_dir=NEURAL_MODEL_CACHE_DIR
            )
            .to(self._device, non_blocking=True)
            .eval()
        )

    def _generate_rewrite(self, input_ids: torch.Tensor) -> str:
        """Generates rewrite given input ids.

        Args:
            input_ids: A sequence of tokens.

        Returns:
            A rewrite.
        """
        # Generate output token ids
        output_ids = self._model.generate(
            input_ids,
            max_length=self._max_length,
            num_beams=self._num_beams,
            early_stopping=self._early_stopping,
        )

        # Decode output token ids
        rewrite = self._tokenizer.decode(
            output_ids[0],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )

        return rewrite

    def _generate_sparse_rewrite(
        self, input_ids: torch.Tensor
    ) -> Dict[str, float]:
        """Generates top-k rewrites and their probabilities where k is
        self._num_beams.

        Args:
            input_ids: A sequence of tokens.

        Returns:
            Rewrites and their probabilities.
        """
        # Generate output
        output = self._model.generate(
            input_ids.to(self._device, non_blocking=True),
            max_length=self._max_length,
            num_beams=self._num_beams,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=self._num_beams,
        )

        # Decode output and return sequences
        scores = torch.exp(output.sequences_scores).tolist()
        factor = sum(scores)
        rewrites = {
            self._tokenizer.decode(
                seq,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            ): score
            / factor
            for seq, score in zip(output.sequences, scores)
        }

        return rewrites

    def rewrite_query(
        self,
        query: Query,
        context: Context = None,
        use_canonical_response: bool = _USE_CANONICAL_RESPONSE,
        index_name: str = _INDEX_NAME,
        separator: str = _SEPARATOR,
        year: str = _YEAR,
    ) -> Union[Query, SparseQuery]:
        """Rewrites query to a new contextualized query given context.

        Based on the training regime of the "castorini/t5-base-canard" model, as
        input, the query and context are concatenated with a separation token
        "|||" that indicates a boundary of queries (and responses) of different
        conversation turns.
        For more details see https://huggingface.co/castorini/t5-base-canard.

        In case of T5 fine-tuned on QReCC a standard <sep> is use as a
        separation token.

        Example:
            Context: "How do you know when your garage door opener is going
                bad?"
            Query: "Now it stopped working. Why?"

            Reformulation:
                Now the garage door opener stopped working. Why?

        Args:
            query: Query to rewrite.
            context (optional): Context containing additional information for
              the rewrite. Defaults to None.
            use_canonical_response: Determines whether canonical responses
              should be used for rewriting the query.
            index_name: The name of the index to be used for loading canonical
              responses.
            separator: Special token to separate input sequences.

        Returns:
            Rewritten query.
        """
        # Nothing to rewrite if there is no context
        if context is None:
            return query

        # Construct input text
        history_questions = [q.question for q, _ in context.history]
        input_text = separator.join(history_questions)
        if use_canonical_response:
            passage_loader = PassageLoader(index=index_name)
            canonical_response = ""
            for doc in context.history[-1][1]:
                canonical_response = canonical_response + passage_loader.get(
                    doc.doc_id
                )
            split_canonical_response = canonical_response.split(" ")
            all_questions_length = len(
                " ".join(history_questions + [query.question]).split(" ")
            )
            if (
                len(split_canonical_response) + all_questions_length
                > _MAX_LENGTH
            ):
                split_position = len(split_canonical_response) - (
                    _MAX_LENGTH - all_questions_length
                )
                canonical_response = " ".join(
                    split_canonical_response[:(-split_position)]
                )
            input_text += f"{separator + canonical_response}"
        input_text += f"{separator + query.question}"

        # Get input token ids
        input_ids = self._tokenizer.encode(
            input_text, return_tensors="pt", add_special_tokens=True
        ).to(self._device)

        if self.sparse:
            rewrites = self._generate_sparse_rewrite(input_ids)
            return SparseQuery(
                query.query_id,
                max(rewrites, key=rewrites.get),
                weighted_match_queries=rewrites,
            )

        rewrite = self._generate_rewrite(input_ids)
        return Query(query.query_id, rewrite)


def rewrite_queries_with_fine_tuned_model(
    model_dir: str,
    year: str,
    max_length: int,
    output_dir: str,
    use_previous_rewritten_utterance: bool,
    use_responses: bool,
    index_name: str,
    separator: str,
    sparse: bool,
    num_beams: int,
):
    """Rewrites queries using a fine-tuned T5 model.

    Args:
        model_dir: The path to the fine-tuned model to be loaded and used for
          rewriting.
        year: Year for which the rewrites should be generated.
        max_length: Max sequence length for the model.
        output_dir: The path to the output directory for generated query
          rewrites.
        use_previous_rewritten_utterance: Specifies whether previously rewritten
          queries should be used in the context for rewriting current query.
        use_responses: Specifies whether canonical responses should be used in
          the context for rewriting current query.
        index_name: The name of the index to be used for loading canonical
          responses.
        separator: Special token to separate input sequences.
    """
    rewriter = T5Rewriter(
        model_name=model_dir,
        max_length=max_length,
        num_beams=num_beams,
        sparse=sparse,
    )
    contexts = Topic.load_contexts_from_file(year, QueryRewrite.AUTOMATIC)
    queries = Topic.load_queries_from_file(year)

    with open(output_dir, "w") as rewrites_out:
        tsv_writer = csv.writer(rewrites_out, delimiter="\t")
        tsv_writer.writerow(
            ["conversation_id", "turn_id", "id", "query", "original", "sparse"]
        )
        rewrites = []
        for query, context in zip(queries, contexts):
            if use_previous_rewritten_utterance and context is not None:
                context.history = [
                    (rewrites[-(len(context.history) - idx)], history[1])
                    for idx, history in enumerate(context.history)
                ]
            rewrite = rewriter.rewrite_query(
                query=query,
                context=context,
                use_canonical_response=use_responses,
                index_name=index_name,
                separator=separator,
                year=year,
            )
            rewrites.append(rewrite)
            tsv_writer.writerow(
                [
                    query.query_id.split("_")[0],
                    query.query_id.split("_")[1],
                    query.query_id,
                    rewrite.question,
                    query.question,
                    json.dumps(rewrite.weighted_match_queries)
                    if isinstance(rewrite, SparseQuery)
                    else "",
                ]
            )


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser(prog="t5_rewriter.py")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=_MODEL_DIR,
        help=(
            "The path to the fine-tuned model to be loaded and used for"
            " rewriting."
        ),
    )
    parser.add_argument(
        "--year",
        type=str,
        default=_YEAR,
        choices=["2020", "2021", "2022"],
        help="Year for which the rewrites should be generated.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=_MAX_LENGTH,
        help="Max sequence length for the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_OUTPUT_DIR,
        help="The path to the output directory for generated query rewrites.",
    )
    parser.add_argument(
        "--use_previous_rewritten_utterance",
        type=bool,
        default=_USE_PREVIOUS_REWRITTEN_UTTERANCE,
        help=(
            "Specifies whether previously rewritten queries should be used in "
            "the context for rewriting current query."
        ),
    )
    parser.add_argument(
        "--use_canonical_response",
        type=bool,
        default=_USE_CANONICAL_RESPONSE,
        help=(
            "Specifies whether the last canonical response should be used in "
            "the context for rewriting current query."
        ),
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=_INDEX_NAME,
        help=(
            "The name of the index to be used for loading canonical responses."
        ),
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=_SEPARATOR,
        help="Special token to separate input sequences.",
    )

    parser.add_argument(
        "--sparse",
        action="store_const",
        const=True,
        help="If true, generates sparse rewrites. Defaults to False.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=_NUM_BEAMS,
        help="Number of beams to use in beam search.",
    )
    return parser.parse_args()


def example_rewriter_usage():
    """Example usage of the rewriter."""
    rewriter = T5Rewriter()
    context = Context()
    prev_query = Query(
        1, "How do you know when your garage door opener is going bad?"
    )
    context.history = [(prev_query, None)]
    rewrite = rewriter.rewrite_query(
        Query(2, "Now it stopped working. Why?"), context
    )
    print(rewrite.question)
    # Should output
    # >>> Now the garage door opener stopped working. Why?


def main(args):
    """Rewrites queries using a fine-tuned model.

    Args:
        args: Arguments.
    """
    rewrite_queries_with_fine_tuned_model(
        model_dir=args.model_dir,
        year=args.year,
        max_length=args.max_length,
        output_dir=args.output_dir,
        use_previous_rewritten_utterance=args.use_previous_rewritten_utterance,
        use_responses=args.use_canonical_response,
        index_name=args.index_name,
        separator=args.separator,
        sparse=args.sparse,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
