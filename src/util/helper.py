import pyconll
import os
import sys

# Setup Working Directory
wdir = os.path.dirname(os.getcwd())
# Add Working Directory to Path
sys.path.append(wdir)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cut_and_convert_to_id(tokens:str, tokenizer, max_input_length:int):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens:str, max_input_length:int):
    """
    Cut tokens to max input length
    :param tokens:
    :param max_input_length:
    :return:
    """
    tokens = tokens[:max_input_length-1]
    return tokens

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

def to_sentence(tagged_list):
    sentence = []
    tags = []
    for token, tag in tagged_list:
        sentence.append(token)
        tags.append(tag)
    return sentence, tags