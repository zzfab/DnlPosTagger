import streamlit as st
import torch
import os
import sys
from transformers import BertTokenizerFast
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.model.bert import PosTaggingModel
from src.util.tagger import id2tag,tag2color,generate_color_legend

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

st.title("POS Tagger")

text = st.text_area("Enter your text here:", "")

st.markdown(generate_color_legend(), unsafe_allow_html=True)

if st.button("Tag"):
    model = PosTaggingModel.load_from_checkpoint("pos_tagging_model.ckpt")
    html_output = predict(text, model)
    st.markdown(html_output, unsafe_allow_html=True)