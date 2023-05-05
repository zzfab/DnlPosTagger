import os
import sys
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import CSVLogger
wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)
from src.util import logger
from src.model.bert import PosTaggingModel
from src.util.helper import *
import settings

logger = logger.setup_applevel_logger(file_name = os.path.join(wdir,'logging/app_debug.log'))
logger.info('Run Trainer for POS Tagging')


def main(test_file):
    model = PosTaggingModel.load_from_checkpoint(os.path.join(wdir,"model/pos_tagging_model.ckpt"),test_file=test_file)
    test_dataloader = model.test_dataloader()
    csv_logger = CSVLogger("lightning_logs", name="pos_tagging")
    trainer = pl.Trainer(accelerator=settings.ACCELERATOR,
                         devices=1,
                         logger=csv_logger)
    trainer.test(model,
                 dataloaders=test_dataloader,
                 verbose=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, nargs="?", default=os.path.join(wdir, "data/UD_English-GUM/en_gum-ud-test.conllu"), help="Path to the test file")
    args = parser.parse_args()
    main(args.test_file)
