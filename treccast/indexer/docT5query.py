import argparse
from typing import List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

_DOC2QUERY_MODEL = "castorini/doc2query-t5-base-msmarco"


class DocT5Query:
    def __init__(self, n_queries: int = 3) -> None:
        """Initializes docT5query for predicting possible queries from text.

        Args:
            n_queries: how many queries to predict per document (defaults
              to 3).
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = T5Tokenizer.from_pretrained(_DOC2QUERY_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(
            _DOC2QUERY_MODEL
        )
        self.model.to(self.device)

        self.n_queries = n_queries

    def predict_queries(self, doc_text: str) -> List[str]:
        """Predicts n_queries for a given document text.

        Args:
            doc_text: Document text.

        Returns:
            List of len self.n_queries where each element is a predicted query
              for given doc_text.
        """

        input_ids = self.tokenizer.encode(doc_text, return_tensors="pt").to(
            self.device
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_k=10,
            num_return_sequences=self.n_queries,
        )

        return [
            self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            for i in range(self.n_queries)
        ]

    def expand_with_queries(self, doc_text: str) -> str:
        """Expands given document text with predicted queries.

        Args:
            doc_text: Text for which to generate queries.

        Returns:
            Original text expanded with queries.
        """
        predicted_queries = " ".join(self.predict_queries(doc_text))
        return f"{doc_text} {predicted_queries}"


def main(args: argparse.Namespace) -> None:
    """Runs docT5query on documents based on commadline arguments.

    Args:
        args: Command line arguments.
    """
    doc_text = """The presence of communication amid scientific minds was
    equally important to the success of the Manhattan Project as scientific
    intellect was. The only cloud hanging over the impressive achievement of the
    atomic researchers and engineers is what their success truly meant; hundreds
    of thousands of innocent lives obliterated."""
    docT5query = DocT5Query(args.n_queries)
    predicted = docT5query.predict_queries(doc_text)
    assert len(predicted) == args.n_queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DocT5Query")
    parser.add_argument(
        "-n",
        "--n_queries",
        type=int,
        default=3,
        help="How many queries to predict for each text.",
    )
    args = parser.parse_args()
    main(args)
