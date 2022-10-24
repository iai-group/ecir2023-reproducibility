"""Query rewriter based on historical context using T5."""

import argparse
import csv
from typing import Union

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from treccast.core import NEURAL_MODEL_CACHE_DIR
from treccast.core.base import Context, Query
from treccast.core.topic import QueryRewrite, Topic
from treccast.rewriter.rewriter import Rewriter

# The path to the fine-tuned model to be loaded and used for rewriting.
_MODEL_DIR = (
    "data/fine_tuning/rewriter/qrecc/T5_QReCC_WaterlooClarke-full/"
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
# Specifies how many last canonical responses should be used in the context
# for rewriting current query.
_USE_CANONICAL_RESPONSES = 1
# Whether to use provided answer rewrites or full passage texts in context.
_USE_ANSWER_REWRITE = False
# Special token to separate input sequences.
_SEPARATOR = "<sep>"
# Number of beams to use in beam search.
_NUM_BEAMS = 10
# Stop beam search when num_beams have completed.
_EARLY_STOPPING = True


class T5Rewriter(Rewriter):
    def __init__(
        self,
        model_name: str = "castorini/t5-base-canard",
        separator: str = _SEPARATOR,
        max_length: int = 64,
        num_beams: int = _NUM_BEAMS,
        early_stopping: bool = _EARLY_STOPPING,
    ) -> None:
        """Instantiate T5 rewriter.

        Args:
            model_name (optional): Hugging Face model name. Defaults to
              "castorini/t5-base-canard".
            separator (optional): Special token to separate input sequences.
            max_length (optional): Max sequence length. Default model was
              trained with sequences up to 64. Defaults to 64.
            num_beams (optional): How many beams to explore. Defaults to 10.
            early_stopping (optional): Whether to stop when num_beams is
              reached. Defaults to True.
        """
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.separator = separator
        self._max_length = max_length
        self._num_beams = num_beams
        self._early_stopping = early_stopping

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

    def rewrite_query(
        self,
        query: Query,
        context: Context = None,
        use_canonical_response: int = _USE_CANONICAL_RESPONSES,
    ) -> Union[Query]:
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

        Returns:
            Rewritten query.
        """
        # Nothing to rewrite if there is no context
        if context is None:
            return query

        # Construct input text
        history_questions = [q.question for q, _ in context.history]
        input_text = self._tokenizer.tokenize(
            self.separator.join(history_questions)
        )
        if use_canonical_response == 1:
            canonical_response = " ".join(
                doc.content for doc in context.history[-1][1]
            )
            split_canonical_response = self._tokenizer.tokenize(
                canonical_response
            )
            all_questions_length = len(
                input_text
                + [self.separator]
                + self._tokenizer.tokenize(query.question)
            )
            if (
                len(split_canonical_response) + all_questions_length
                > _MAX_LENGTH
            ):
                split_position = len(split_canonical_response) - (
                    _MAX_LENGTH - all_questions_length
                )
                split_canonical_response = split_canonical_response[
                    :(-split_position)
                ]
            input_text += [self.separator] + split_canonical_response
        elif use_canonical_response == 3:
            canonical_response = " ".join(
                doc.content for doc in context.history[-1][1]
            )
            if len(context.history) > 1:
                canonical_response += " ".join(
                    doc.content for doc in context.history[-2][1]
                )
                if len(context.history) > 2:
                    canonical_response += " ".join(
                        doc.content for doc in context.history[-3][1]
                    )
            input_text += [self.separator] + self._tokenizer.tokenize(
                canonical_response
            )
        input_text += [self.separator] + self._tokenizer.tokenize(
            query.question
        )

        # Get input token ids
        input_ids = self._tokenizer.encode(
            input_text, return_tensors="pt", add_special_tokens=True
        ).to(self._device)

        rewrite = self._generate_rewrite(input_ids)
        return Query(query.query_id, rewrite)


def rewrite_queries_with_fine_tuned_model(
    model_dir: str,
    year: str,
    max_length: int,
    output_dir: str,
    use_previous_rewritten_utterance: bool,
    use_responses: bool,
    use_answer_rewrite: bool,
    use_canonical_response: int,
    separator: str,
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
        use_answer_rewrite: If true, the document content is the rewritten
          answer, otherwise its full passage(s). Defaults to False.
        separator: Special token to separate input sequences.
        num_beams: Number of beams to use in beam search.
    """
    rewriter = T5Rewriter(
        model_name=model_dir,
        separator=separator,
        max_length=max_length,
        num_beams=num_beams,
    )
    contexts = Topic.load_contexts_from_file(
        year,
        QueryRewrite.AUTOMATIC,
        use_answer_rewrite,
    )
    queries = Topic.load_queries_from_file(year)

    with open(output_dir, "w") as rewrites_out:
        tsv_writer = csv.writer(rewrites_out, delimiter="\t")
        tsv_writer.writerow(
            [
                "conversation_id",
                "turn_id",
                "id",
                "query",
                "original",
            ]
        )
        rewrites = []
        for query, context in zip(queries, contexts):
            if use_previous_rewritten_utterance and context is not None:
                if use_canonical_response == 1:
                    context.history = [
                        (rewrites[-(len(context.history) - idx)], history[1])
                        for idx, history in enumerate(context.history)
                    ]
            rewrite = rewriter.rewrite_query(
                query=query,
                context=context,
                use_canonical_response=use_responses,
            )
            rewrites.append(rewrite)
            tsv_writer.writerow(
                [
                    query.query_id.split("_")[0],
                    query.query_id.split("_")[1],
                    query.query_id,
                    rewrite.question,
                    query.question,
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
        choices=["2020", "2021"],
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
        type=int,
        default=_USE_CANONICAL_RESPONSES,
        help=(
            "Specifies whether the last canonical response should be used in "
            "the context for rewriting current query."
        ),
    )
    parser.add_argument(
        "--use_answer_rewrite",
        type=bool,
        default=_USE_ANSWER_REWRITE,
        help=(
            "Specifies whether to use provided answer rewrites or full passage "
            "texts in context."
        ),
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=_SEPARATOR,
        help="Special token to separate input sequences.",
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
        use_answer_rewrite=args.use_answer_rewrite,
        separator=args.separator,
        num_beams=args.num_beams,
        use_canonical_response=args.use_canonical_response,
    )


if __name__ == "__main__":
    args = parse_cmdline_arguments()
    main(args)
