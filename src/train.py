import os
import sys
import pytorch_lightning as pl
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.util import logger
from src.model.bert import PosTaggingModel
from src.util.helper import *
import settings
import argparse
from pytorch_lightning.loggers import CSVLogger

logger = logger.setup_applevel_logger(file_name = os.path.join(wdir,'logging/app_debug.log'))
logger.info('Run Trainer for POS Tagging')

def main(train_file,dev_file):
    model = PosTaggingModel(train_file=train_file,
                            dev_file=dev_file)
    csv_logger = CSVLogger("lightning_logs", name="pos_tagging")
    trainer = pl.Trainer(max_epochs=settings.EPOCH,
                         accelerator=settings.ACCELERATOR,
                         devices=3,
                         logger=csv_logger)
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(wdir,"model/pos_tagging_model.ckpt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, nargs="?", default=os.path.join(wdir, "data/UD_English-GUM/en_gum-ud-train.conllu"), help="Path to the training file")
    parser.add_argument("dev_file", type=str, nargs="?", default=os.path.join(wdir, "data/UD_English-GUM/en_gum-ud-dev.conllu"), help="Path to the development file")
    args = parser.parse_args()

    main(args.train_file, args.dev_file)