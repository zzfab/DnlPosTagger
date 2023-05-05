# POS Tagger
This repository contains a Part-of-Speech (POS) tagger implemented using the BERTForTokenClassification model from the Hugging Face Transformers library.

Code was written in Python 3.8.16
Be aware of change `settings.py` if computation is made on cpu/gpu
## Assumptions
- The input data is in CoNLL-U format.
- The dataset has 17 distinct POS tags.

## Model Choice Reasoning
BERT (Bidirectional Encoder Representations from Transformers) is chosen for this task because it has proven to be highly effective in a variety of natural language processing tasks, including POS tagging. The pre-trained model can be fine-tuned to the specific task, which allows it to leverage the knowledge it has already gained during pre-training on a large corpus.

BERTForTokenClassification is a BERT model specifically designed for token-level tasks like POS tagging. It extends the base BERT model by adding a classification layer on top of the hidden states for each token. This makes it suitable for the given task.

## Testing Strategy and Confidence in the Model
The testing strategy involves splitting the dataset into training, validation, and test sets. The model is trained on the training set and evaluated on the validation set during training to monitor its performance and prevent overfitting. Finally, the model is tested on the test set to obtain its performance on unseen data.

The degree of confidence in the model can be determined by comparing its performance on the validation and test sets. If the model performs well on both sets, it indicates that it generalizes well to unseen data. However, the model's confidence can be further improved by conducting a more thorough evaluation, such as using cross-validation or testing on a diverse set of datasets.

## Trade-offs
- BERT models are computationally expensive and may require more resources (e.g., memory and computation time) compared to LSTM or HMM models. If you have limited resources, you may want to consider simpler models like LSTMs or HMMs.
- Using a pre-trained BERT model like bert-base-uncased may not be optimal for certain languages or specific domains. Fine-tuning a domain-specific pre-trained model or training a model from scratch could be beneficial in such cases.
- HMMs are more interpretable than BERT or LSTM models, which might be an advantage in some applications where model interpretability is crucial.
- No testing/hyperparametertuning/early stopping or regularization was made because of time
## Time Spent on Challenge
(BERT) approach took around ~ 7h in total.

## Streamlit Web App
The Streamlit web app provides an interactive interface for data analysis, evaluation, and generation of POS tags. It consists of three main sections:

1. Data Analyzer: Displays various plots and statistics for the training and validation datasets.
2. Evaluation: Displays plots for training and validation losses for each epoch, as well as evaluation metrics for the test set, such as accuracy and F1-score.
3. Generator: Allows users to input their own text and generates POS tags for the input text using the trained model.

To launch the Streamlit web app, run:
```
cd app/
streamlit run data_analyzer.py
```

## Usage:
To train the model, run: 
``` 
python train.py --train_file [train_file] --dev_file [dev_file] 
``` 
where train_file is the path to the training data and dev_file is the path to the development data. The trained model will be saved to disk as model.pickle.

To test the accuracy of the model, run:
``` 
python eval.py --test_file [test_file]
``` 
where test_file is the path to the test data.

To generate POS tags for unlabelled text, 
```
run python generate.py --text_file [text_file]
```
where text_file is the path to the unlabelled text.


