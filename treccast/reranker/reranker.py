"""Abstract interface for a reranker."""

from abc import ABC, abstractmethod
from typing import List, Tuple
from numpy import ndarray

from treccast.core.query.query import Query
from treccast.core.ranking import Ranking

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Reranker(ABC):
    def __init__(self) -> None:
        """Interface for a reranker."""
        pass

    @abstractmethod
    def rerank(self, query: Query, ranking: Ranking) -> Ranking:
        """Performs reranking.

        Returns:
            New Ranking instance with updated scores.
        """
        raise NotImplementedError


class NeuralReranker(Reranker):
    def __init__(
        self,
        model_dir: str = "nboost/pt-bert-base-uncased-msmarco",
        max_seq_len: int = 256,
    ) -> None:
        """Neural reranker. Currently only supports BERT type architecture.

        Args:
            model_dir (optional): Location to the model. Defaults to
                "nboost/pt-bert-base-uncased-msmarco".
            max_seq_len (optional): Maximal number of tokens. Defaults
                to 256.
        """
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._max_seq_len = max_seq_len

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        )

        self._model.to(self._device, non_blocking=True)

    def rerank(
        self,
        query: Query,
        ranking: Ranking,
    ) -> Ranking:
        """Returns new ranking with updated scores from the neural reranker.

        Args:
            query: Query for which to re-rank.
            ranking: Current rankings for the query.

        Returns:
            Ranking containing new scores for each document.
        """
        doc_ids, documents = ranking.documents()
        logits = self._get_logits(query.question, documents)

        # Note: logit[0] corresponds to the document not being relevant and
        # logit[1] corresponds to the document being relevant.
        return Ranking(
            ranking.query_id,
            [
                {"doc_id": doc_id, "score": logit[1]}
                for (logit, doc_id) in zip(logits, doc_ids)
            ],
        )

    def _get_logits(self, query: str, documents: List[str]) -> ndarray:
        """Returns logits from the neural model.

        Args:
            query: Query for which to evaluate.
            documents: List of documents to evaluate.

        Returns:
            Numpy array containing two values for each document: the probability
                of the document being non-relevant [0] and relevant [1].
        """
        # TODO Split into manageable batch sizes.
        # https://github.com/iai-group/trec-cast-2021/issues/66
        input_ids, attention_mask, token_type_ids = self._encode(
            query, documents
        )

        with torch.no_grad():
            logits = self._model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0]
            logits = logits.detach().cpu().numpy()

            return logits

    def _encode(self, query: str, documents: List[str]) -> Batch:
        """Tokenize and collate a number of single inputs, adding special
        tokens and padding.

        Returns:
            Batch: Input IDs, attention masks, token type IDs
        """
        inputs = self._tokenizer.batch_encode_plus(
            [[query, document] for document in documents],
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=self._max_seq_len,
        )

        input_ids = torch.tensor(inputs["input_ids"]).to(
            self._device, non_blocking=True
        )
        attention_mask = torch.tensor(inputs["attention_mask"]).to(
            self._device, non_blocking=True
        )
        token_type_ids = torch.tensor(inputs["token_type_ids"]).to(
            self._device, non_blocking=True
        )

        return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    ranking1 = Ranking("qid_0")
    ranking1.add_doc(
        "1",
        50.62,
        "Many people search for âstandard garage door sizesâ on a daily "
        "basis. However there are many common size garage door widths and"
        "heights but the standard size is probably more a matter of the age"
        "of your home and what area of the town, state, or country that you "
        "live in. There are a number of standard sizes for residential garage "
        "doors in the United States.",
    )
    ranking1.add_doc(
        "2",
        1.52,
        "The presence of communication amid scientific minds was equally"
        "important to the success of the Manhattan Project as scientific"
        "intellect was. The only cloud hanging over the impressive achievement"
        " of the atomic researchers and engineers is what their success truly "
        "meant; hundreds of thousands of innocent lives obliterated.",
    )
    ranking1.add_doc(
        "3",
        80.22,
        "Garage Door Opener Problems. So, when the garage door opener decides "
        "to take a day off, it can leave you stuck outside, probably during a "
        "rain or snow storm. Though they may seem complicated, there really are"
        " several things most homeowners can do to diagnose and repair opener "
        "failures.nd, if you are careful not to damage the door or the seal on "
        "the bottom of the door, use a flat shovel or similar tool to chip away"
        "at the ice. Once you get the door open, clear any water, ice or snow "
        "from the spot on the garage floor where the door rests when closed",
    )

    ranking2 = Ranking("qid_1")
    ranking2.add_doc(
        "1",
        20.43,
        "Typically, it will cost less to install a steel garage door without an"
        " opener than to install a custom wood door with a garage door opener. "
        "Recent innovations have also yielded high-tech doors with thick "
        "insulation and energy-efficient glaze, as well as finished interior "
        "surfaces and other significant upgrades.f your garage door has started"
        " to malfunction, you might be considering installing a new or upgraded"
        " door. Rest assured it is a smart investment. In fact, installing a "
        "new garage door yields about 84 percent in resale value, according to "
        "Remodeling Magazine",
    )
    ranking2.add_doc(
        "4",
        12.3,
        "Organize volunteer community panels, boards, or committees that meet "
        "with the offender to discuss the incident and offender obligation to "
        "repair the harm to victims and community members. Facilitate the "
        "process of apologies to victims and communities. Invite local victim "
        "advocates to provide ongoing victim-awareness training for probation "
        "staff",
    )
    ranking2.add_doc(
        "5",
        100,
        "Purchasing extra remotes and getting openers set up for operation will"
        " typically range from $100 to $400, which will add to the overall cost"
        " of the garage door installation. If your opener works with the new "
        "door, you won't need to have it replaced. In cases in which the new "
        "door is much heavier than the old door, however, the old garage door "
        "opener won't be able to handle the extra weight. This is something to "
        "keep in mind when you're shopping for a new garage door.",
    )

    query1 = Query(
        "qid_0",
        "How do you know when your garage door opener is going bad?",
    )
    query2 = Query(
        "qid_1",
        "How much does it cost for someone to repair a garage door opener?",
    )
    rankings = [ranking1, ranking2]

    ranker = NeuralReranker()
    print(ranker.rerank(query1, ranking1).fetch_topk_docs())
