import numpy as np
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datasets import SequenceDataModule
from models import SequenceTransformerModule
from config import parser

if __name__ == '__main__':
    hparams = parser.parse_args()

    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    logger = WandbLogger(name=hparams.run_name,
                         version=datetime.now().strftime('%y%m%d_%H%M%S.%f'),
                         project='Trans2D',
                         config=hparams)

    datamodule = SequenceDataModule(dataset_name=hparams.dataset_name,
                                    max_seq_len=hparams.max_position_embeddings,
                                    batch_size=hparams.batch_size,
                                    num_workers=hparams.num_workers)
    datamodule.prepare_data()
    hparams.vocab = datamodule.vocab
    hparams.n_attributes = datamodule.n_attributes

    model = SequenceTransformerModule(hparams)

    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         num_sanity_val_steps=0,
                         log_every_n_steps=1)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)