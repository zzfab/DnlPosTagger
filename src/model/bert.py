import os
import pytorch_lightning as pl
import sys
# Setup Working Directory
wdir = os.path.dirname(os.getcwd())
# Add Working Directory to Path
sys.path.append(wdir)
print(wdir)
# Import Custom Modules
from db_client.datasets.gmu_dataset import PosTaggingDataset
from src.util import logger
from torch.utils.data import DataLoader
from src.util.helper import *
logger = logger.get_logger(__name__)
import torch
from transformers import BertForTokenClassification,  BertConfig,BertTokenizerFast


class PosTaggingModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=17):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = BertForTokenClassification.from_pretrained(model_name, config=config)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.batch_size = 4

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, tags = batch
        outputs = self(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=-1)
        correct = torch.sum(preds == tags, dtype=torch.float)
        total = preds.numel()
        return {"correct": correct, "total": total}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def validation_dataloader(self,dev_file):
        sentences, tags = load_data(dev_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        return data_loader

    def train_dataloader(self,train_file):
        sentences, tags = load_data(train_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        return data_loader

    def test_dataloader(self,test_file):
        sentences, tags = load_data(test_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        return data_loader
    def test_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log("test_accuracy", accuracy)
        return accuracy

    def compute_accuracy(self,outputs):
        total_correct = sum([output["correct"] for output in outputs])
        total_predictions = sum([output["total"] for output in outputs])
        return total_correct / total_predictions
