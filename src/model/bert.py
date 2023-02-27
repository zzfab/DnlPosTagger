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
from src.model.bertclassifier import BertClassifier

logger = logger.get_logger(__name__)

class BERTPoSTagger(BertClassifier):
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
        #embedding_dim = self.config['text']['sentence_level']['hidden_size']
        self.dropout = nn.Dropout(self.config['dropout'])
        self.fc = nn.Linear(768, self.config['output_dim'])

    def forward(self, x):
        # Get the output of the BERT model
        outputs = self.model(x['input_ids'].squeeze(1), x['attention_mask'])
        pooled_output = outputs[1]
        # Apply dropout and pass the output through the linear layer
        logits = self.fc(self.dropout(pooled_output))

        return logits
