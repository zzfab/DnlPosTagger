import torch
import torch.nn as nn
from typing import Optional
from torchmetrics import MetricCollection
import os
import sys
from transformers import BertModel

# Setup Working Directory
wdir = os.path.dirname(os.getcwd())
# Add Working Directory to Path
sys.path.append(wdir)
print(wdir)
# Import Custom Modules
from src.util import logger
from src.model.classifier import Classifier

logger = logger.get_logger(__name__)


class BERTPoSTagger(Classifier):
    def __init__(
        self,
        config: dict,
        model: Optional[nn.Module] = None,
        metric_collection: Optional[MetricCollection] = None,
    ):
        super(BERTPoSTagger, self).__init__(
            config=config, model=model, metric_collection=metric_collection
        )
        #self.config = self.config, self.model = model, self.metric_collection = metric_collection
        self.model = BertModel.from_pretrained(self.config['text']['pretrained_model'])
        embedding_dim = self.config['text']['sentence_level']['hidden_size']
        self.fc = nn.Linear(768, self.config['output_dim'])
        self.dropout = nn.Dropout(self.config['dropout'])

    def forward(self, x):
        with torch.no_grad():
            x = self.model(**x).pooler_output
        return self.fc(x)