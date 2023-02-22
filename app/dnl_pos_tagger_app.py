import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
import torch


wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)


from src.model.bert import BERTPoSTagger
from src.train import train
from src.util.helper import *

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title('DNL Part of Speech Tagger')


option = st.selectbox(
        "Select a model to analyse",
        ("train", "dev", "test"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled)

data = read_conllu(os.path.join(wdir,f"data/UD_English-GUM/en_gum-ud-{option}.conllu"))
st.write((data[0][0]))

X = []  # store input sequence
Y = []  # store output sequence

for sentence in data:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:
        X_sentence.append(entity[0])  # entity[0] contains the word
        Y_sentence.append(entity[1])  # entity[1] contains corresponding tag

    X.append(X_sentence)
    Y.append(Y_sentence)

num_words = len(set([word.lower() for sentence in X for word in sentence]))
num_tags = len(set([word.lower() for sentence in Y for word in sentence]))

df = pd.DataFrame(zip(X,Y), columns=['Sentence','Tags'])
df['sentence_lenght'] = df['Sentence'].apply(lambda x: len(x))
df['tags_lenght'] = df['Tags'].apply(lambda x: len(x))

#st.write('sample X: ', X[0], '\n')
#st.write('sample Y: ', Y[0], '\n')

st.write("Total number of tagged sentences: {}".format(len(X)))
st.write("Vocabulary size: {}".format(num_words))
st.write("Total number of tags: {}".format(num_tags))

# check length of longest sentence
lengths = [len(seq) for seq in X]
st.write("Length of longest sentence: {}".format(max(lengths)))
st.subheader("Sample Data")
st.dataframe(df.head(10))
#st.dataframe(df.query('Sentence.len() != Tags.len()'))

tag_list = Y
tag_counts = defaultdict(int)

for tags in tag_list:
    for tag in tags:
        tag_counts[tag] += 1
st.json(dict(tag_counts))

