import torch
from torch.utils.data import Dataset
import os
import sys
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.util import logger,tagger
logger = logger.get_logger(__name__)



class PosTaggingDataset(Dataset):
    """
    Dataset class for Part-of-Speech (POS) tagging task.
    """
    def __init__(self, sentences, tags, tokenizer):
        """
       Initialize the dataset with sentences, tags, and tokenizer.

       :param sentences: List of sentences (strings)
       :param tags: List of lists containing POS tags for each sentence
       :param tokenizer: Tokenizer from Hugging Face Transformers library
       """
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: Number of samples in the dataset
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.

        :param idx: Index of the sample
        :return: Dictionary containing input_ids, attention_mask, and labels for the sample
        """
        sentence = self.sentences[idx]
        tag = self.tags[idx]
        # Tokenize the input sentence and tags
        tokenized_input = self.tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True,
                                         padding='max_length', truncation=True)
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        offsets_mapping = tokenized_input['offset_mapping']

        # Map the tags to the tokenized input
        aligned_tags = tagger.align_tags(tag, offsets_mapping)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(aligned_tags)
        }
