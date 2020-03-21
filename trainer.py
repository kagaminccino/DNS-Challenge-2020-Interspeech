"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from argparse import ArgumentParser
from tasks import DNS
#from test_tube import HyperOptArgumentParser


def main(hparams):
    # init module
    model = DNS(hparams)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_pesq',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.checkpoint_path,
        save_best_only=True,
        verbose=True,
        monitor='avg_val_pesq',
        mode='max',
        prefix='pesq'
    )

    tt_logger = TestTubeLogger(
        save_dir=hparams.default_save_path,
        name='logs',
        debug=False
    )
        
    # most basic trainer, uses good defaults
    trainer = Trainer(
        weights_summary=hparams.weights_summary,
        max_nb_epochs=hparams.max_nb_epochs,
        min_nb_epochs=hparams.min_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        default_save_path=hparams.default_save_path,
        gradient_clip_val=hparams.gradient_clip_val,
        track_grad_norm=hparams.track_grad_norm,
        log_save_interval=hparams.log_save_interval,
        train_percent_check=0.16666,
        val_percent_check=1.0,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        # checkpoint_callback=checkpoint_callback,
        checkpoint_callback=False,
        early_stop_callback=early_stop_callback,
        logger=tt_logger
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--mulaw', action='store_true')
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = DNS.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
