import os
import sys

import pytorch_lightning as pl
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.util import logger
from src.model.bert import PosTaggingModel
from src.util.helper import *
logger = logger.setup_applevel_logger(file_name = os.path.join(wdir,'logging/app_debug.log'))
logger.info('Run Application for POS Tagging')


def main():
    # Set up argument parser
    #parser = argparse.ArgumentParser(description='Train POS tagger')
    #parser.add_argument('--train-file', type=str, required=True, help='Path to training data file')
    #parser.add_argument('--dev-file', type=str, required=True, help='Path to development data file')
    #parser.add_argument('--model-file', type=str, default='model.pickle', help='Path to save trained model')
    #parser.add_argument('--hidden-size', type=int, default=128, help='Size of hidden layer in model')
    #parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for training')
    #parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    # Parse arguments
    #args = parser.parse_args()

    # Call train function with arguments
    #ToDo implement train function
    #train.train_model(args.train_file, args.dev_file, args.model_file, args.hidden_size, args.learning_rate, args.epochs)
    return None
if __name__ == '__main__':
    #cfg = {}
    #cfg['train'] = yaml.safe_load(Path(os.path.join(wdir, "src/config/train.yaml")).read_text())
    #cfg['model'] = yaml.safe_load(Path(os.path.join(wdir, "src/config/model.yaml")).read_text())
    #model = PosTaggingModel()
    #train(cfg['train'],model)


    model = PosTaggingModel()

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=3)
    trainer.fit(model)
    trainer.save_checkpoint("pos_tagging_model.ckpt")

    new_model = PosTaggingModel.load_from_checkpoint("pos_tagging_model.ckpt")
    trainer.test(model)