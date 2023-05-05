# POS Tagger
This repository contains a Part-of-Speech (POS) tagger implemented using the BERTForTokenClassification model from the Hugging Face Transformers library.

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
- Using a pre-trained BERT model like bert-base-uncased may not be optimal for certain languages or specific domains. Fine-tuning a domain-specific pre-trained model or training a model from scratch could be beneficial in such cases.
- This task is easy to solve but time consuming to optimize. Therefore iam using a relative simple approach which could be further improved by optimization techniques


## Time Spent on Challenge
Starting with a Hidden Markov approach in the past ~ 10h.
Improved solution for this task with a pre-trained Language Model (BERT) ~ 5h.

## Usage:
To train the model, run: 
``` 
python train.py --train_file [train_file] --dev_file [dev_file] --model_path [model_path_to_save]
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


## Testing Strategy and Confidence
ToDo: 

## Time Spent
ToDo:

## Evaluation Criteria
ToDo: 
