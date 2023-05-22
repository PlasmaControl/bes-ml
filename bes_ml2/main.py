from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from datetime import datetime
import shutil

import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
import wandb

try:
    from . import elm_datamodule
    from . import elm_lightning_model
except:
    from bes_ml2 import elm_datamodule
    from bes_ml2 import elm_lightning_model


@dataclasses.dataclass(eq=False)
class BES_Trainer:
    lightning_model: elm_lightning_model.Lightning_Model
    datamodule: elm_datamodule.ELM_Datamodule
    experiment_dir: str = './experiment_default'
    trial_name: str = None  # if None, use default Tensorboard scheme
    log_freq: int = 100
    wandb_log: bool = False

    def __post_init__(self):

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        if not self.trial_name:
            datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
            if slurm_identifier:
                self.trial_name = f"r{slurm_identifier}_{datetime_str}"
            else:
                self.trial_name = f"r{datetime_str}"

        self.experiment_dir = Path(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = self.experiment_dir.name
        self.experiment_parent_dir = self.experiment_dir.parent

        # set loggers
        tb_logger = TensorBoardLogger(
            save_dir=self.experiment_parent_dir,
            name=self.experiment_name,
            version=self.trial_name,
            default_hp_metric=False,
        )
        self.trial_dir = Path(tb_logger.log_dir).absolute()
        print(f"Trial directory: {self.trial_dir}")
        self.loggers = [tb_logger]

        if self.wandb_log:
            wandb.login()
            wandb_logger = WandbLogger(
                save_dir=self.experiment_dir,
                project=self.experiment_name,
                name=self.trial_name,
            )
            wandb_logger.watch(
                self.lightning_model, 
                log='all', 
                log_freq=self.log_freq,
            )
            self.loggers.append(wandb_logger)

        print("Model Summary:")
        print(ModelSummary(self.lightning_model, max_depth=-1))

    def run_all(
        self,
        max_epochs: int = 2,
        skip_test: bool = False,
        skip_predict: bool = False,
        early_stopping_min_delta: float = 1e-3,
        early_stopping_patience: int = 50,
        gradient_clip_value: int = None,
        float_precision: str|int = '16-mixed' if torch.cuda.is_available() else 32,
    ):
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir
        monitor_metric = self.lightning_model.monitor_metric
        metric_mode = 'min' if 'loss' in monitor_metric else 'max'
        torch.set_float32_matmul_precision('medium')

        # set callbacks
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor=monitor_metric,
                mode=metric_mode,
                save_last=True,
            ),
            EarlyStopping(
                monitor=monitor_metric,
                mode=metric_mode,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
                log_rank_zero_only=True,
                verbose=True,
            ),
        ]

        trainer = Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_value,
            logger=self.loggers,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
            log_every_n_steps=self.log_freq,
            num_nodes=int(os.getenv('SLURM_NNODES', default=1)),
            precision=float_precision,
            strategy=DDPStrategy(find_unused_parameters=True)
        )
        self.datamodule.is_global_zero = trainer.is_global_zero
        if trainer.is_global_zero:
            self.trial_dir.mkdir(parents=True, exist_ok=True)

        trainer.fit(
            self.lightning_model, 
            datamodule=self.datamodule,
        )
        
        if skip_test is False:
            trainer.test(datamodule=self.datamodule, ckpt_path='best')

        if skip_predict is False:
            trainer.predict(datamodule=self.datamodule, ckpt_path='best')

        self.last_model_path = Path(trainer.checkpoint_callback.last_model_path).absolute()
        print(f"Last model path: {self.last_model_path}")
        best_model_path = Path(trainer.checkpoint_callback.best_model_path).absolute()
        self.best_model_path = best_model_path.parent/'best.ckpt'
        shutil.copyfile(
            src=best_model_path,
            dst=self.best_model_path,
        )
        print(f"Best model path: {self.best_model_path}")


if __name__=='__main__':

    checkpoint = None
    # checkpoint = '/global/u2/d/drsmith/ml/bes-ml/bes_ml2/experiment_default/r2023-05-15_12-26-21/checkpoints/last.ckpt'

    if checkpoint:
        # load data and model from checkpoint
        lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
        datamodule = elm_datamodule.ELM_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        # initiate new data and model
        lightning_model = elm_lightning_model.Lightning_Model(
            signal_window_size=64,
            cnn_nlayers=6,
            cnn_num_kernels=4,
            cnn_kernel_time_size=2,
            cnn_padding=[[0,1,1]]*3 + [0]*3,
        )
        datamodule = elm_datamodule.ELM_Datamodule(
            signal_window_size=lightning_model.signal_window_size,
            # max_elms=5,
            batch_size=16,
            # fraction_test=0,
            # fraction_validation=0.2,
        )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        wandb_log=True,
    )

    trainer.run_all(
        max_epochs=6,
        # skip_test=True,
        skip_predict=True,
    )
