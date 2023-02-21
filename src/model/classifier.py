import os
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from abc import ABC, abstractmethod
from typing import Optional
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import F1Score as F1
from torchmetrics import MetricCollection, Precision, Recall
import sys

wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)

from src.util import logger
from src.util.helper import *
from db_client.datasets.gmu_dataset import GMU
logger = logger.get_logger(__name__)

class Classifier(pl.LightningModule,ABC):
    def __init__(
            self,
            config: dict,
            model: Optional[nn.Module] = None,
            metric_collection: Optional[MetricCollection] = None,
    ):
        super(Classifier, self).__init__()
        self.config = config
        self.model = None or model
        self.metric_collection = metric_collection or MetricCollection(
            [
                Accuracy(num_classes=self.config['num_classes'], task="multiclass"),
                Precision(num_classes=self.config['num_classes'], average="macro",task="multiclass"),
                Recall(num_classes=self.config['num_classes'], average="macro",task="multiclass"),
                F1(num_classes=self.config['num_classes'], threshold=0.5, average="weighted",task="multiclass"),
            ]
        )
        self.train_metrics = self.metric_collection.clone(prefix="train/")
        self.val_metrics = self.metric_collection.clone(prefix="val/")

    @abstractmethod
    def forward(self, x):
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learnring_rate)
        return optimizer
    def log_metrics(
        self,
        batch_idx: int,
        logits: torch.Tensor,
        label: torch.Tensor,
        mode: str,
        on_epoch: bool = True,
        enable_logger: bool = True,
    ):
        if batch_idx % 1000 == 0:
            _, pred = torch.max(logits, 1)
            logger.debug(f"Labels: {label} | Pred: {pred}")

        for k, v in self.train_metrics(logits, label).items():
            self.log(f"{mode}/{k}", v, on_epoch=on_epoch, logger=enable_logger)

    def training_step(self, batch, batch_idx):
        data, label = batch

        text = data
        tags = label
        logits = self(text)
        logits = logits.view(-1, logits.shape[-1])
        tags = tags
        loss = self.ce_criterion(logits, tags)
        self.log(
            self.config['train_step']['name'],
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(
            batch_idx=batch_idx,
            logits=logits,
            label=tags,
            mode="train",
            on_epoch=True,
            enable_logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        text = data
        tags = label
        logits = self(text)
        logits = logits.view(-1, logits.shape[-1])
        tags = tags
        loss = self.ce_criterion(logits, tags)
        self.log(
            self.config['val_step']['name'],
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(
            batch_idx=batch_idx,
            logits=logits,
            label=tags,
            mode="val",
            on_epoch=True,
            enable_logger=True,
        )
        return loss

    def ce_criterion(self, pred, target, weight=None, reduction="mean"):
        return cross_entropy(pred, target, weight=weight, reduction=reduction)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=float(self.config['optim']['lr']),
            weight_decay=float(self.config['optim']['weight_decay']),
            amsgrad=self.config['optim']['amsgrad'],
        )
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    opt,
                    patience=self.config['scheduler']['patience'],
                    verbose=self.config['scheduler']['verbose'],
                    factor=self.config['scheduler']['factor'],
                    min_lr=self.config['scheduler']['min_lr'],
                ),
                "monitor": self.config['scheduler']['monitor'],
                "interval": self.config['scheduler']['interval'],
                "frequency": self.config['scheduler']['frequency'],
            }
        ]
        return [opt], schedulers

    def text_collate_fn(self,batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prepare text input data and labels to send into the BERT network
        :param batch: list of tuples containing data and label, [(data1, label1), ... (data2, label2)]
        :return: tuple containing stacked tensors for data and label
        """
        texts, labels = [], []
        for item in batch:
            text, lab = item
            texts.append(text)
            labels.append(lab)
        text = texts[0]
        text.data["input_ids"] = torch.cat([t.data["input_ids"] for t in texts])
        # text.data["token_type_ids"] = torch.cat([t.data["token_type_ids"] for t in texts]) # for BertTokenizer, not in use
        text.data["attention_mask"] = torch.cat([t.data["attention_mask"] for t in texts])
        return text, torch.flatten(torch.tensor(labels))

    def train_dataloader(self):
        train_lists = read_conllu(os.path.join(wdir,"data/UD_English-GUM/en_gum-ud-train.conllu"))
        sentences = []
        tags = []
        for sentence in train_lists:
            s, t = to_sentence(sentence)
            sentences.append(s)
            tags.append(t)

        train_dataloader = DataLoader(GMU(sentences,tags),batch_size=4,collate_fn=self.text_collate_fn)
        return train_dataloader

    def validation_dataloader(self):
        val_lists = read_conllu(os.path.join(wdir,"data/UD_English-GUM/en_gum-ud-dev.conllu"))
        sentences = []
        tags = []
        for sentence in val_lists:
            s, t = to_sentence(sentence)
            sentences.append(s)
            tags.append(t)
        val_dataloader = DataLoader(GMU(sentences,tags),batch_size=4,collate_fn=self.text_collate_fn)
        return val_dataloader


