import streamlit as st
import torch
import os
import sys
from transformers import BertTokenizerFast
import argparse
import pytorch_lightning as pl
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.model.bert import PosTaggingModel
from src.util.tagger import id2tag,tag2color,generate_color_legend
import settings

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def predict(text, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, axis=-1).squeeze().tolist()
        tags = [id2tag[id] for id in preds]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        # Remove [CLS] and [SEP] tokens
        tokens = tokens[1:-1]
        tags = tags[1:-1]
        # Build the HTML string
        html_output = ""
        for token, tag in zip(tokens, tags):
            style = tag2color[tag]
            html_output += f'<span style="{style}">{token}</span> '
        return html_output

def main(test_file):
    model = PosTaggingModel.load_from_checkpoint(os.path.join(wdir,"model/pos_tagging_model.ckpt"),test_file=test_file)
    with open(args.text_file, "r") as f:
        text = f.readlines()
    sentences = text.split("\n")
    html_outputs = predict(sentences, model)
    for html_output in html_outputs:
        print(html_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str, nargs="?", help="Path to the text file")
    args = parser.parse_args()
    main(args.text_file)
