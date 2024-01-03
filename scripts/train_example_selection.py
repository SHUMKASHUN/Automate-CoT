import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

import sys


from src.example_selection import ExampleSelection

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int,default=1)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--entity", type=str, default="test")
    parser.add_argument("--enable_wandb", action='store_true')

    parser = ExampleSelection.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        seed_everything(seed=args.seed)

    logger = WandbLogger(project=args.project_name, name=args.run_name, entity=args.entity) if args.enable_wandb else None

    callbacks = [
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(
        args, 
        logger=logger if logger else True, 
        callbacks=callbacks,
        max_epochs = 5,
        #strategy = DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps = 0,
        val_check_interval = 1.0
    )

    model = ExampleSelection(**vars(args))

    trainer.fit(model)
