import argparse
from typing import List, Any, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import (
    AdamW,
    get_constant_schedule_with_warmup,
    AutoModel,
    AutoTokenizer,
)
import sys
import os

from treccast.retrieval.reranker.reranker import Reranker
from treccast.core.query.query import Query
from treccast.core.ranking import Ranking
from treccast.retrieval.reranker.pytorch_dataset import (
    PointWiseDataset,
    Batch,
    TrainBatch,
)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12555"


class BERTReranker(Reranker, LightningModule):
    def __init__(
        self,
        queries: List[Query],
        rankings: List[Ranking],
        hparams: Dict[str, Any],
    ) -> None:
        """Vanilla BERT re-ranker.

        Args:
            queries: List of queries to be re-ranked.
            ranking: List of rankings for the queries.
            hparams: Dictionary of parameters used for training and evaluating
                the models.
        """
        Reranker.__init__(self, rankings=rankings, queries=queries)
        LightningModule.__init__(self)
        self._hp = hparams
        self._bert_model = AutoModel.from_pretrained(
            self._hp.get("bert_type"), return_dict=True
        )
        self._dropout = torch.nn.Dropout(self._hp.get("dropout"))
        self._classification = torch.nn.Linear(self._hp["bert_dim"], 1)
        for p in self._bert_model.parameters():
            p.requires_grad = not hparams["freeze_bert"]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hp.get("bert_type")
        )
        # TODO right now self._dataset is used for training, testing and
        # predicting, create a separate dataset objects when training and
        # predicting is needed separately.
        self._dataset = PointWiseDataset(
            queries=queries,
            rankings=rankings,
            tokenizer=self._tokenizer,
            max_len=self._hp.get("max_len"),
        )
        self._batch_size = self._hp["batch_size"]
        self._num_workers = self._hp.get("num_workers")
        self._uses_ddp = "ddp" in self._hp.get("accelerator", "")
        self._batch_size = self._hp.get("batch_size")
        self._bce = torch.nn.BCEWithLogitsLoss()

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
        if self._uses_ddp:
            sampler = DistributedSampler(self._dataset, shuffle=True)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(self._dataset, "collate_fn", None),
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

    def test_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function
        named `collate_fn`, it is used. If the model is trained in DDP mode,
        the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """
        if self._uses_ddp:
            sampler = DistributedSampler(self._dataset, shuffle=True)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(self._dataset, "collate_fn", None),
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

    def predict_dataloader(self) -> DataLoader:
        """Return a testset DataLoader if the testset exists.
        If the testset object has a function named `collate_fn`, it is used.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no testing
                dataset.
        """

        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=getattr(self._dataset, "collate_fn", None),
        )

    def predict_step(
        self, batch: TrainBatch, batch_idx: int, dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Predict a single batch. The returned input is tokenized query,
        doc pairs

        Args:
            batch: inputs and labels
            batch_idx: Batch index (not used)
            dataloader_idx: DataLoader index (not used)

        Returns:
            Dict[str, torch.Tensor]: predictions and gold labels
        """
        inputs, labels = batch
        pred_dict = {"prediction": self(inputs), "label": labels}
        return pred_dict

    @staticmethod
    def get_lightning_trainer(ap):
        trainer = Trainer.from_argparse_args(
            ap,
            deterministic=True,
            replace_sampler_ddp=False,
            default_root_dir="out",
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
            default=1,
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
            "--training_mode",
            choices=["pointwise", "pairwise"],
            default="pairwise",
            help="Training mode",
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
            default="out",
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
    queries = [
        Query(
            "qid_0",
            "How do you know when your garage door opener is going bad?",
        ),
        Query(
            "qid_1",
            "How much does it cost for someone to repair a garage door opener?",
        ),
    ]
    ranking1 = Ranking("0")
    ranking1.add_doc(
        "1",
        "Many people search for âstandard garage door sizesâ on a daily "
        "basis. However there are many common size garage door widths and"
        "heights but the standard size is probably more a matter of the age"
        "of your home and what area of the town, state, or country that you "
        "live in. There are a number of standard sizes for residential garage "
        "doors in the United States.",
        50.62,
    )
    ranking1.add_doc(
        "2",
        "The presence of communication amid scientific minds was equally"
        "important to the success of the Manhattan Project as scientific"
        "intellect was. The only cloud hanging over the impressive achievement"
        " of the atomic researchers and engineers is what their success truly "
        "meant; hundreds of thousands of innocent lives obliterated.",
        1.52,
    )
    ranking1.add_doc(
        "3",
        "Garage Door Opener Problems. So, when the garage door opener decides "
        "to take a day off, it can leave you stuck outside, probably during a "
        "rain or snow storm. Though they may seem complicated, there really are"
        " several things most homeowners can do to diagnose and repair opener "
        "failures.nd, if you are careful not to damage the door or the seal on "
        "the bottom of the door, use a flat shovel or similar tool to chip away"
        "at the ice. Once you get the door open, clear any water, ice or snow "
        "from the spot on the garage floor where the door rests when closed",
        80.22,
    )

    ranking2 = Ranking("1")
    ranking2.add_doc(
        "1",
        "Typically, it will cost less to install a steel garage door without an"
        " opener than to install a custom wood door with a garage door opener. "
        "Recent innovations have also yielded high-tech doors with thick "
        "insulation and energy-efficient glaze, as well as finished interior "
        "surfaces and other significant upgrades.f your garage door has started"
        " to malfunction, you might be considering installing a new or upgraded"
        " door. Rest assured it is a smart investment. In fact, installing a "
        "new garage door yields about 84 percent in resale value, according to "
        "Remodeling Magazine",
        20.43,
    )
    ranking2.add_doc(
        "4",
        "Organize volunteer community panels, boards, or committees that meet "
        "with the offender to discuss the incident and offender obligation to "
        "repair the harm to victims and community members. Facilitate the "
        "process of apologies to victims and communities. Invite local victim "
        "advocates to provide ongoing victim-awareness training for probation "
        "staff",
        12.3,
    )
    ranking2.add_doc(
        "5",
        "Purchasing extra remotes and getting openers set up for operation will"
        " typically range from $100 to $400, which will add to the overall cost"
        " of the garage door installation. If your opener works with the new "
        "door, you won't need to have it replaced. In cases in which the new "
        "door is much heavier than the old door, however, the old garage door "
        "opener won't be able to handle the extra weight. This is something to "
        "keep in mind when you're shopping for a new garage door.",
        100,
    )

    # Uncomment this if you want to use multi-gpu training
    # dist.init_process_group("gloo", rank=1, world_size=1)
    rankings = [ranking1, ranking2]
    ap = BERTReranker.add_model_specific_args()
    # Change the bert_type to something which pretrained for ms-marco passage
    # ranking for example, this huggingface model
    # "bert_type=nboost/pt-bert-base-uncased-msmarco"".
    # By default it uses bert-base-cased.
    ap_dict = ap.parse_args().__dict__
    print(ap_dict["bert_type"])
    # Create a pytorch-lightning trainer with all the training arguments
    trainer = BERTReranker.get_lightning_trainer(ap)
    # Create a BERT ranker which has a linear classification head on top of BERT
    bert_reranker = BERTReranker(queries, rankings, ap_dict)
    # trainer.fit trains the model by calling the train_dataloader and
    #  training_step
    trainer.fit(bert_reranker)
    # trainer.test test the model by calling the test_dataloader and
    #  testing_step
    print(trainer.test())
    # trainer.predict test the model by calling the predict_dataloader and
    #  predict_step
    print(trainer.predict())
