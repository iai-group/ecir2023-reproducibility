"""Query rewriter based on historical context using T5."""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from treccast.core import NEURAL_MODEL_CACHE_DIR
from treccast.core.base import Context, Query
from treccast.rewriter.rewriter import Rewriter


class T5Rewriter(Rewriter):
    def __init__(
        self,
        model_name: str = "castorini/t5-base-canard",
        max_length: int = 64,
        num_beams: int = 10,
        early_stopping: bool = True,
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
        """
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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

    def rewrite_query(
        self,
        query: Query,
        context: Context = None,
    ) -> Query:
        """Rewrites query to a new contextualized query given context.

        Based on the training regime of the "castorini/t5-base-canard" model, as
        input, the query and context are concatenated with a separation token
        "|||" that indicates a boundary of queries (and responses) of different
        conversation turns.
        For more details see https://huggingface.co/castorini/t5-base-canard.

        Example:
            Context: "How do you know when your garage door opener is going bad?"
            Query: "Now it stopped working. Why?"

            Reformulation:
                Now the garage door opener stopped working. Why?

        Args:
            query: Query to rewrite.
            context (optional): Context containing additional information for
                the rewrite. Defaults to None.

        Returns:
            Rewritten query.
        """
        # Nothing to rewrite if there is no context
        if not context:
            return query

        # Construct input text
        # Currently uses only past queries and not responses
        input_text = " ||| ".join(q.question for q, _ in context.history)
        input_text += f" ||| {query.question}"

        # Get input token ids
        input_ids = self._tokenizer.encode(
            input_text, return_tensors="pt", add_special_tokens=True
        ).to(self._device)

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

        return Query(query.query_id, rewrite)


if __name__ == "__main__":
    # Example usage
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
