from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import typing
import dataclasses
from datetime import datetime, timedelta
import time
import psutil, os, sys

# 3rd-party imports
import yaml
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
import pytorch_lightning.callbacks


# @dataclasses.dataclass(eq=False)
# class Train_Base_Dataclass:
#     output_dir: Path | str = Path('run_dir')  # path to output dir.
#     n_epochs: int = 2  # training epochs
#     learning_rate: float = 1e-3  # optimizer learning rate


# @dataclasses.dataclass(eq=False)
# class Train_Base(Train_Base_Dataclass):

#     def __post_init__(self):
#         t_start_setup = time.time()


@dataclasses.dataclass(eq=False)
class Model_Base_Dataclass:
    output_dir: Path | str = Path('run_dir')  # path to output dir.
    n_epochs: int = 2  # training epochs
    learning_rate: float = 1e-3  # optimizer learning rate
    signal_window_size: int = 64  # power of 2; ~16-512
    dropout_rate: float = 0.1  # ~0.1


@dataclasses.dataclass(eq=False)
class Model_Base(pl.LightningModule, Model_Base_Dataclass):

    def __post_init__(self):
        super().__init__()  # nn.Module.__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = F.cross_entropy(y_hat, y)
    #     self.log("test_loss", test_loss)

    # def predict_step(self, batch, batch_idx):
    #     x, _ = batch
    #     pred = self(x)
    #     return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class DataModule_Base(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # self.mnist_predict = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        pass

    def train_dataloader(self):
        return (
            torch.randn(size=(32,1,28,28)),
            torch.randint(low=0, high=9, size=(32,1))
        )
        # return torch.utils.data.DataLoader(
        #     [
        #         torch.randn(size=(1,28,28)),
        #         torch.randint(low=0, high=9, size=(1,))
        #     ], 
        #     batch_size=self.batch_size,
        # )

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            [torch.randn(size=(32,1,28,28)),], 
            batch_size=self.batch_size,
        )


if __name__=='__main__':
    model = Model_Base()
    trainer = pl.Trainer(
        default_root_dir='./run_dir',
        num_sanity_val_steps=0,
        limit_train_batches=None,
        limit_val_batches=None,
        max_time=timedelta(seconds=30),
        # fast_dev_run=True,
        # profiler='simple',
        callbacks=[
            # pytorch_lightning.callbacks.DeviceStatsMonitor(cpu_stats=True),
        ],
    )
    trainer.fit(
        model=model,
        datamodule=DataModule_Base(),
    )