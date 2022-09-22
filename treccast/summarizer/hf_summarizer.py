"""Passage summarizer based on a Hugging Face model."""

from transformers import pipeline
from treccast.core.base import ScoredDocument
from treccast.core.ranking import Ranking
from treccast.summarizer.summarizer import Summarizer

_DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"


class HuggingFaceSummarizer(Summarizer):
    def __init__(self, model_name: str = _DEFAULT_SUMMARIZER_MODEL) -> None:
        """Instantiates a summarizer based on a Hugging Face model.

        The summarization is based on the Pipelines API developed by Hugging
        Face.
        For more details see:
        https://huggingface.co/docs/transformers/v4.20.0/en/main_classes/pipelines#transformers.SummarizationPipeline

        Args:
            model_name (optional): Hugging Face model name. Defaults to
              file-level constant _DEFAULT_SUMMARIZER_MODEL.
        """  # noqa
        self._summarizer = pipeline("summarization", model=model_name)

    def summarize_passages(
        self,
        passages: Ranking,
        k: int = 3,
        min_length: int = 10,
        max_length: int = 250,
    ) -> str:
        """Summarizes passages using a Hugging Face model.

        Args:
            passages: Passages to summarize.
            k (optional): Maximum number of passages to consider for the
              summary.
            min_length (optional): Minimum number of tokens in the summary.
              Defaults to 10 tokens.
            max_length (optional): Maximum number of tokens in the summary.
              Defaults to 250 tokens.

        Returns:
            Abstractive summary of passages.
        """
        topk = passages.fetch_topk_docs(k=k, unique=True)
        texts = list(map(lambda p: p.content, topk))
        text = " ".join(texts)
        summary = self._summarizer(
            text, min_length=min_length, max_length=max_length
        )
        return summary[0]["summary_text"]


if __name__ == "__main__":
    # Example usage
    summarizer = HuggingFaceSummarizer()

    passages = Ranking(
        "qid_0",
        [
            ScoredDocument(
                doc_id="1",
                content=(
                    "Many people search for ’standard garage door sizes’ "
                    "on a daily basis. However there are many common size "
                    "garage door widths and heights but the standard size is "
                    "probably more a matter of the age of your home and what "
                    "area of the town, state, or country that you live in. "
                    "There are a number of standard sizes for residential "
                    "garage doors in the United States."
                ),
                score=50.62,
            ),
            ScoredDocument(
                doc_id="2",
                content=(
                    "The presence of communication amid scientific minds was "
                    "equally important to the success of the Manhattan Project "
                    "as scientific intellect was. The only cloud hanging over "
                    "the impressive achievement of the atomic researchers and "
                    "engineers is what their success truly meant; hundreds of "
                    "thousands of innocent lives obliterated."
                ),
                score=1.52,
            ),
            ScoredDocument(
                doc_id="3",
                content=(
                    "Garage Door Opener Problems. So, when the garage door "
                    "opener decides to take a day off, it can leave you stuck "
                    "outside, probably during a rain or snow storm. Though "
                    "they may seem complicated, there really are several "
                    "things most homeowners can do to diagnose and repair "
                    "opener failures.nd, if you are careful not to damage the "
                    "door or the seal on the bottom of the door, use a flat "
                    "shovel or similar tool to chip away at the ice. Once you "
                    "get the door open, clear any water, ice or snow from the "
                    "spot on the garage floor where the door rests when closed"
                ),
                score=80.22,
            ),
        ],
    )

    summary = summarizer.summarize_passages(passages)
    print(summary)
    # Should output
    # >>> There are a number of standard sizes for residential garage doors in
    # the United States. The presence of communication amid scientific minds
    # was equally important to the success of the Manhattan Project as
    # scientific intellect was.
