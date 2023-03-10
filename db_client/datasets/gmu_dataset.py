from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import  BertTokenizer
import os
import sys
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)

from src.util import logger
logger = logger.get_logger(__name__)


class GMU(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mapper = {
            '<pad>': 0, 'NOUN': 1, 'PUNCT': 2, 'VERB': 3, 'PRON': 4, 'ADP': 5, 'DET': 6, 'PROPN': 7, 'ADJ': 8,
             'AUX': 9, 'ADV': 10, 'CCONJ': 11, 'PART': 12, 'NUM': 13, 'SCONJ': 14, 'X': 15, 'INTJ': 16, 'SYM': 17
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        txt = " ".join(self.sentences[idx])
        logger.debug(f"encoded_text: {txt}")
        encoded_text = self.tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        encoded_labels = [self.mapper[x] for x in self.tags[idx]]
        return encoded_text,torch.tensor(encoded_labels)

