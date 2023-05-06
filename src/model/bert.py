import os
import pytorch_lightning as pl
import sys
# Setup Working Directory
wdir = os.path.dirname(os.getcwd())
# Add Working Directory to Path
sys.path.append(wdir)
print(wdir)
# Import Custom Modules
from db_client.datasets.gum_dataset import PosTaggingDataset
from src.util import logger
from src.util.tagger import id2tag
from torch.utils.data import DataLoader
from src.util.helper import *
import settings
logger = logger.get_logger(__name__)
import torch
from transformers import BertForTokenClassification,  BertConfig,BertTokenizerFast
import torchmetrics
class PosTaggingModel(pl.LightningModule):
    """
    PyTorch Lightning module for Part-of-Speech tagging using BERTForTokenClassification.
    """
    def __init__(self, train_file=None, dev_file=None, test_file=None, model_name="bert-base-uncased", batch_size=settings.BATCH_SIZE, num_labels=17):
        """
        Initialize the model, tokenizer, and other configurations.
        """
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = BertForTokenClassification.from_pretrained(model_name, config=config)

        self.batch_size = batch_size

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss,on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
         Perform a single validation step.
         """
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.
        """
        input_ids, attention_mask, tags = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=tags)
        preds = torch.argmax(outputs.logits, axis=-1)

        # Filter out the -100 values from both preds and tags tensors
        valid_positions = tags != -100
        filtered_preds = preds[valid_positions]
        filtered_tags = tags[valid_positions]

        self.test_acc(filtered_preds, filtered_tags)  # Pass filtered preds and tags
        self.log('test_acc', self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def val_dataloader(self):
        sentences, tags = load_data(self.dev_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        return data_loader

    def train_dataloader(self):
        sentences, tags = load_data(self.train_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        return data_loader

    def test_dataloader(self):
        sentences, tags = load_data(self.test_file)
        dataset = PosTaggingDataset(sentences, tags, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        return data_loader

