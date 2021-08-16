import argparse
import os
import sys

from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics import (
    MetricCollection,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)
from transformers import (
    AdamW,
    get_constant_schedule_with_warmup,
    AutoModel,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Any, Tuple, Dict, Sequence, Optional

from treccast.core.query.query import Query
from treccast.core.ranking import Ranking
from treccast.core.util.load_finetune_data import FineTuneDataLoader
from treccast.reranker.bert_reranker import BERTReranker
from treccast.reranker.train.pytorch_dataset import (
    PointWiseDataset,
    Batch,
    TrainBatch,
)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12556"
os.environ["WORLD_SIZE"] = "4"
os.environ["RANK"] = "0"


class BERTRerankTrainer(LightningModule):
    def __init__(
        self,
        train_data: Tuple[List[Query], List[Ranking]],
        hparams: Dict[str, Any],
        val_data: Optional[Tuple[List[Query], List[Ranking]]] = None,
        test_data: Optional[Tuple[List[Query], List[Ranking]]] = None,
    ) -> None:
        """Vanilla BERT re-ranker.

        Args:
            train_data: Tuple of parallel list of queries and rankings to train.
            train_data: Same type as train_data but used for validation.
            test_data: Same type as train_data but used for data.
            hparams: Dictionary of parameters used for training and evaluating
                the models.
        """
        LightningModule.__init__(self)
        # This line is needed initialize the DDP but it seems to hang.
        # Use DP instead.
        # dist.init_process_group(backend="nccl")
        self._hp = hparams
        self._bert_model = AutoModel.from_pretrained(
            self._hp.get("bert_type"), return_dict=True
        )
        self._dropout = torch.nn.Dropout(self._hp.get("dropout"))
        self._classification = torch.nn.Linear(self._hp["bert_dim"], 1)
        print(hparams["freeze_bert"])
        for p in self._bert_model.parameters():
            p.requires_grad = not hparams["freeze_bert"]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hp.get("bert_type")
        )
        # Use part of the training data for validation

        self._train_dataset = PointWiseDataset(
            queries=train_data[0],
            rankings=train_data[1],
            tokenizer=self._tokenizer,
            max_len=self._hp.get("max_len"),
        )

        self._val_dataset = (
            PointWiseDataset(
                queries=val_data[0],
                rankings=val_data[1],
                tokenizer=self._tokenizer,
                max_len=self._hp.get("max_len"),
            )
            if val_data
            else None
        )

        self._test_dataset = (
            PointWiseDataset(
                queries=test_data[0],
                rankings=test_data[1],
                tokenizer=self._tokenizer,
                max_len=self._hp.get("max_len"),
            )
            if test_data
            else None
        )

        self._batch_size = self._hp["batch_size"]
        self._num_workers = self._hp.get("num_workers")
        self._uses_ddp = "ddp" in self._hp.get("accelerator", "")
        self._batch_size = self._hp.get("batch_size")
        self._bce = torch.nn.BCEWithLogitsLoss()
        self._val_metrics = MetricCollection(
            [
                RetrievalMAP(compute_on_step=False),
                RetrievalMRR(compute_on_step=False),
                RetrievalNormalizedDCG(compute_on_step=False),
            ],
            prefix="val_",
        )
        if self._uses_ddp:
            self._sampler = DistributedSampler(
                self._train_dataset, shuffle=True
            )
            self._shuffle = None
        else:
            self._sampler = None
            self._shuffle = True

    @property
    def val_metric_names(self) -> Sequence[str]:
        """Return all validation metrics that are computed after each epoch.
        Returns:
            Sequence[str]: The metric names
        """
        return self._val_metrics.keys()

    def forward(self, batch: Batch) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch: BERT inputs

        Returns:
            torch.Tensor: The output scores, shape (batch_size, 1)
        """
        cls_out = self._bert_model(*batch)["last_hidden_state"][:, 0]
        return self._classification(self._dropout(cls_out))

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.

        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler
        """
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(params_with_grad, lr=self._hp["lr"])
        sched = get_constant_schedule_with_warmup(opt, self._hp["warmup_steps"])
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function
        named `collate_fn`, it is used. If the model is trained in DDP mode,
        the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(self._train_dataset, "collate_fn", None),
        )

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        """Train a single batch.

        Args:
            batch: A training batch, depending on
            the mode batch_idx: Batch index

        Returns:
            torch.Tensor: Training loss
        """

        inputs, labels = batch
        loss = self._bce(self(inputs).flatten(), labels.flatten())
        self.log("train_loss", loss)
        return loss

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a trainset DataLoader. If the trainset object has a function
        named `collate_fn`, it is used. If the model is trained in DDP mode,
        the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(self._val_dataset, "collate_fn", None),
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Return a testset DataLoader if the testset exists.
        If the testset object has a function named `collate_fn`, it is used.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no testing
                dataset.
        """
        if self._test_dataset is None:
            return None
        else:
            return DataLoader(
                self._test_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=getattr(self._test_dataset, "collate_fn", None),
            )

    def predict_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        """Train a single batch.

        Args:
            batch: A training batch, depending on
            the mode batch_idx: Batch index (not used)

        Returns:
            torch.Tensor: Training loss
        """

        inputs, _ = batch
        return self(inputs)

    def test_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function
        named `collate_fn`, it is used. If the model is trained in DDP mode,
        the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """

        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(self._test_dataset, "collate_fn", None),
        )

    def test_step(self, batch: TrainBatch, batch_idx: int):
        """Process a single test batch.

        Args:
            batch: inputs and labels
            batch_idx: Batch index (not used)
        """
        inputs, labels = batch
        out_dict = {
            "tokens": [input for input in inputs[0]],
            "attention": [input for input in inputs[1]],
            "prediction": self(inputs),
            "label": labels,
        }
        return out_dict

    def save_model(self, folder: str):
        self._bert_model.save_pretrained(folder)

    @staticmethod
    def get_lightning_trainer(ap):
        trainer = Trainer.from_argparse_args(
            ap,
            deterministic=True,
            replace_sampler_ddp=False,
            default_root_dir="data/local/out",
        )
        return trainer

    @staticmethod
    def add_model_specific_args():
        """Add model-specific arguments to the parser.

        Returns:
            ap: The argument parser
        """
        ap = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # trainer args
        # Trainer.add_argparse_args would make the help too cluttered
        ap.add_argument(
            "--accumulate_grad_batches",
            type=int,
            default=1,
            help="Update weights after this many batches",
        )
        ap.add_argument(
            "--max_epochs",
            type=int,
            default=10,
            help="Maximum number of epochs",
        )
        ap.add_argument(
            "--gpus", type=int, nargs="+", help="GPU IDs to train on"
        )
        ap.add_argument(
            "--val_check_interval",
            type=float,
            default=1.0,
            help="Validation check interval",
        )
        ap.add_argument(
            "--save_top_k", type=int, default=1, help="Save top-k checkpoints"
        )
        ap.add_argument(
            "--limit_val_batches",
            type=int,
            default=sys.maxsize,
            help="Use a subset of validation data",
        )
        ap.add_argument(
            "--limit_train_batches",
            type=int,
            default=sys.maxsize,
            help="Use a subset of training data",
        )
        ap.add_argument(
            "--limit_test_batches",
            type=int,
            default=sys.maxsize,
            help="Use a subset of test data",
        )
        ap.add_argument(
            "--precision",
            type=int,
            choices=[16, 32],
            default=32,
            help="Floating point precision",
        )
        ap.add_argument(
            "--accelerator",
            default="",
            help="Distributed backend (accelerator) ddp",
        )

        # model args
        ap.add_argument(
            "--bert_type", default="bert-base-uncased", help="BERT model"
        )
        ap.add_argument(
            "--bert_dim", type=int, default=768, help="BERT output dimension"
        )
        ap.add_argument(
            "--dropout", type=float, default=0.1, help="Dropout percentage"
        )
        ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
        ap.add_argument(
            "--loss_margin",
            type=float,
            default=0.2,
            help="Margin for pairwise loss",
        )
        ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
        ap.add_argument(
            "--warmup_steps",
            type=int,
            default=1000,
            help="Number of warmup steps",
        )
        ap.add_argument(
            "--freeze_bert",
            action="store_true",
            help="Do not update any weights of BERT (only train the \
                classification layer)",
        )
        ap.add_argument(
            "--num_workers",
            type=int,
            default=16,
            help="Number of DataLoader workers",
        )

        # remaining args
        ap.add_argument(
            "--val_patience", type=int, default=3, help="Validation patience"
        )
        ap.add_argument(
            "--save_dir",
            default="data/local/out",
            help="Directory for logs, checkpoints and predictions",
        )
        ap.add_argument(
            "--random_seed", type=int, default=123, help="Random seed"
        )
        ap.add_argument(
            "--load_weights", help="Load pre-trained weights before training"
        )
        ap.add_argument(
            "--test", action="store_true", help="Test the model after training"
        )
        return ap


if __name__ == "__main__":
    seed_everything(7)
    fnt_loader = FineTuneDataLoader()
    queries, rankings = fnt_loader.get_query_ranking_pairs()
    ap = BERTRerankTrainer.add_model_specific_args()
    # Change the bert_type to something which pretrained for ms-marco passage
    # ranking for example, this huggingface model
    # "bert_type=nboost/pt-bert-base-uncased-msmarco"".
    # By default it uses bert-base-cased.
    ap_dict = ap.parse_args().__dict__

    # Create a pytorch-lightning trainer with all the training arguments
    trainer = BERTRerankTrainer.get_lightning_trainer(ap)
    ap_dict["bert_type"] = "nboost/pt-bert-base-uncased-msmarco"
    print(ap_dict["bert_type"])

    # Create a BERT ranker which has a linear classification head on top of BERT
    bert_reranker = BERTRerankTrainer((queries, rankings), ap_dict)
    # trainer.fit trains the model by calling the train_dataloader and
    #  training_step

    trainer.fit(bert_reranker)
    folder = f'{ap_dict["bert_type"]}_fine_tuned'
    bert_reranker.save_model(folder)

    # trainer.test test the model by calling the test_dataloader and
    #  testing_step if calling test() make sure test data is passed in init.
    # print(trainer.test())
    # trainer.predict test the model by calling the predict_dataloader and
    #  predict_step if calling predict() make sure test data is passed in init.
    # print(trainer.predict())

    # Validate the trained ranker by using it in BERTReranker
    ranker = BERTReranker(folder)
    print(
        ranker.rerank(query=queries[0], ranking=rankings[0]).fetch_topk_docs()
    )