import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.util.helper import read_conllu


class GMU(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.mapper = {
            '<pad>': 0, 'NOUN': 1, 'PUNCT': 2, 'VERB': 3, 'PRON': 4, 'ADP': 5, 'DET': 6, 'PROPN': 7, 'ADJ': 8,
             'AUX': 9, 'ADV': 10, 'CCONJ': 11, 'PART': 12, 'NUM': 13, 'SCONJ': 14, 'X': 15, 'INTJ': 16, 'SYM': 17
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        print(self.tags[idx])
        encoded_text = self.tokenizer.encode_plus(
            " ".join(self.sentences[idx]),
            add_special_tokens=True,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        labels = [self.mapper[x] for x in self.tags[idx]]
        print(labels)
        return encoded_text,labels

