

from transformers import BertTokenizer, BertModel
import os
import sys
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Setup Working Directory
wdir = os.getcwd()
# Add Working Directory to Path
sys.path.append(wdir)

# Import Custom Modules
from src.model.bert import BERTPoSTagger
from src.util import logger
from src.util.helper import count_parameters


log = logger.get_logger(__name__)



def setup_callbacks(cfg: dict):
    """Setup callbacks for training."""
    early_stop_callback = EarlyStopping(
        monitor=cfg['val_step']['name'],
        min_delta=cfg['early_stop']['min_delta'],
        patience=cfg['early_stop']['patience'],
        verbose=True,
        mode=cfg['early_stop']['mode'],
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['val_step']['name'],
        save_top_k=cfg['save_top_k_epochs'],
        save_last=True,
        mode="min",
    )
    return [early_stop_callback, checkpoint_callback]

def train(cfg: dict, model: LightningModule):
    """Call to trainer module of PyTorch Lightning, trains model."""
    trainer = Trainer(
        callbacks=setup_callbacks(cfg),
        devices=cfg['devices'],
        accelerator=cfg['accelerator'],
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        max_epochs=cfg['epochs'],
        log_every_n_steps=cfg['log_every_n_steps'],
        min_epochs=cfg['min_epochs'],
        resume_from_checkpoint=cfg['checkpoint'],
        fast_dev_run=True,
    )
    trainer.fit(model)

