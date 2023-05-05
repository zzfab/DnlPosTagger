import pyconll
import os
import sys
import torch
# Setup Working Directory
wdir = os.path.dirname(os.getcwd())
# Add Working Directory to Path
sys.path.append(wdir)

def read_conllu(path):
    """
    Read conllu file
    :param path: path to conllu file
    :return: tagged sentences
    """
    data = pyconll.load_from_file(path)
    tagged_sentences=[]
    t=0
    for sentence in data:
        tagged_sentence=[]
        for token in sentence:
            if token.upos and token.form:
                t+=1
                tagged_sentence.append((token.form.lower(), token.upos))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences

def to_sentence(tagged_sentence):
    words = [token[0] for token in tagged_sentence]
    tags = [token[1] for token in tagged_sentence]
    return words, tags


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
def load_data(file_path):
    data = read_conllu(file_path)
    sentences = []
    tags = []

    for sentence in data:
        s, t = to_sentence(sentence)
        sentences.append(s)
        tags.append(t)
    return sentences, tags