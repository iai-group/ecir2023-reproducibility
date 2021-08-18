from typing import List

import torch

from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.train.bert_reranker_train import BERTRerankTrainer


class BERTRerankerFinetuned(BERTReranker):
    def __init__(
        self,
        max_seq_len: int = 512,
        batch_size: int = 512,
        checkpoint_path="/data/scratch/trec-cast-2021/data/models/fine_tuned_models/epoch=2-step=3236.ckpt",
    ) -> None:
        """This class predicts the relevance for query, document pairs using
        model in the specified checkpoint.

        Args:
            max_seq_len: Max sequence length. Defaults to 512.
            batch_size: Prediction batch size. Defaults to 32.
            checkpoint_path: Path to load checkpoint from.
        """
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size)
        ckpt = checkpoint_path
        ap = BERTRerankTrainer.add_model_specific_args()
        args_parsed, _ = ap.parse_known_args()
        ap_dict = args_parsed.__dict__
        ap_dict["bert_type"] = "nboost/pt-bert-base-uncased-msmarco"
        self._batch_size = batch_size
        kwargs = {
            "training_mode": None,
            "rr_k": None,
            "num_workers": None,
            "freeze_bert": True,
            "train_data": None,
            "hparams": ap_dict,
        }
        self._model = BERTRerankTrainer.load_from_checkpoint(ckpt, **kwargs)
        self._model.to(self._device)
        self._model.eval()
        self._sigmoid = torch.nn.Sigmoid()

    def _get_logits(
        self, query: str, documents: List[str]
    ) -> List[List[float]]:
        """Returns logits from the neural model.

        Args:
            query: Query for which to evaluate.
            documents: List of documents to evaluate.

        Returns:
            A list containing two values for each document: the probability
                of the document being non-relevant [0] and relevant [1].
        """
        input_ids, attention_mask, token_type_ids = self._encode(
            query, documents
        )

        with torch.no_grad():
            outputs = self._model((input_ids, attention_mask, token_type_ids))
            probability = self._sigmoid(outputs).tolist()
        return [[1 - result[0], result[0]] for result in probability]
