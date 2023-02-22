# POS Tagger
this repository contains a POS tagger that can be trained on labeled data and used to generate tags for unlabelled text. The following scripts are provided for this purpose:
```
train.py: trains the model on labeled data, with one sentence per line and each token separated by a space.
eval.py: tests the accuracy of the model on labeled test data, with the same format as the training data.
generate.py: generates POS tags for unlabelled text, with one sentence per line and each token separated by a space.
```
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
