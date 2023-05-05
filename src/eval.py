import os
import sys
import pytorch_lightning as pl
import argparse
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.util import logger
from src.model.bert import PosTaggingModel
from src.util.helper import *
import settings

logger = logger.setup_applevel_logger(file_name = os.path.join(wdir,'logging/app_debug.log'))
logger.info('Run Trainer for POS Tagging')


def main(test_file):
    model = PosTaggingModel.load_from_checkpoint("pos_tagging_model.ckpt")
    trainer = pl.Trainer(accelerator=settings.ACCELERATOR)
    trainer.test(model, test_file=test_file, verbose=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, nargs="?", default=os.path.join(wdir, "data/UD_English-GUM/en_gum-ud-test.conllu"), help="Path to the test file")
    args = parser.parse_args()
    main(args.test_file)
