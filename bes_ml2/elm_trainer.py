from __future__ import annotations
import dataclasses
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import callbacks as cb
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

import elm_data
import elm_model


@dataclasses.dataclass(eq=False)
class BES_Trainer:
    experiment_group_dir: str = './Experiment_test'
    experiment_name: str = None  # if None, use default Tensorboard scheme
    max_epochs: int = 2
    gradient_clip_value: int = None
    wandb_log: bool = False
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 10
    enable_progress_bar: bool = True
    num_nodes: int = 1
    datamodule: pl.LightningDataModule = None
    model: pl.LightningModule = None
    monitor: str = 'val_score'
    wandb_log_freq: int = 100
    pl_log_freq: int = 50

    def __post_init__(self):
        assert self.datamodule and self.model
        if self.wandb_log:
            wandb.login()
        self.trainer = None

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            if field_name in ['datamodule', 'model']:
                continue
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        print("Model Summary:")
        print(ModelSummary(self.model, max_depth=-1))

    def run_fast_dev(self):
        tmp_trainer = pl.Trainer(
            fast_dev_run=True,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
        )
        tmp_trainer.fit(
            model=self.model, 
            datamodule=self.datamodule,
        )

    def make_loggers_and_callbacks(self):
        self.experiment_group_dir = os.path.abspath(self.experiment_group_dir)
        os.makedirs(self.experiment_group_dir, exist_ok=True)

        # set loggers
        self.loggers = []
        tb_logger = loggers.TensorBoardLogger(
            save_dir=os.path.dirname(self.experiment_group_dir), 
            name=os.path.basename(self.experiment_group_dir), 
            version=self.experiment_name,
        )
        self.loggers.append(tb_logger)
        version_str = tb_logger.version if tb_logger.version is str else f"version_{tb_logger.version}"
        if hasattr(self.model, 'log_dir'):
            self.model.log_dir = tb_logger.log_dir
        
        if self.wandb_log:
            wandb_logger = loggers.WandbLogger(
                save_dir=self.experiment_group_dir,
                project=os.path.basename(self.experiment_group_dir),
                name=version_str,
            )
            wandb_logger.watch(
                self.model, 
                log='all', 
                log_freq=self.wandb_log_freq,
            )
            self.loggers.append(wandb_logger)

        # set callbacks
        self.callbacks = [
            cb.LearningRateMonitor(),
            cb.EarlyStopping(
                monitor=self.monitor,
                mode='min' if 'loss' in self.monitor else 'max',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
            ),
        ]

    def run_all(self):
        self.make_loggers_and_callbacks()

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=self.loggers,
            callbacks=self.callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.pl_log_freq,
            num_nodes=self.num_nodes,
            accelerator="auto",
            devices="auto",
        )

        self.trainer.fit(self.model, datamodule=self.datamodule)
        self.trainer.test(datamodule=self.datamodule, ckpt_path='best')
        self.trainer.predict(datamodule=self.datamodule, ckpt_path='best')


if __name__=='__main__':
    signal_window_size = 128
    datamodule = elm_data.ELM_Datamodule(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        signal_window_size=signal_window_size,
        max_elms=100,
        batch_size=128,
        fraction_validation=0.1,
        fraction_test=0.1,
    )
    model = elm_model.Lit_Model_CNN01(
        signal_window_size=signal_window_size,
        lr=1e-3,
        weight_decay=1e-5,
    )
    trainer = BES_Trainer(
        model=model,
        datamodule=datamodule,
        max_epochs=2,
        wandb_log=False,
    )
    trainer.run_all()
