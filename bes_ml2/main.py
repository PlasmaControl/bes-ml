from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from datetime import datetime

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
    from . import elm_torch_model
except:
    from bes_ml2 import elm_datamodule
    from bes_ml2 import elm_lightning_model
    from bes_ml2 import elm_torch_model




@dataclasses.dataclass(eq=False)
class BES_Trainer:
    lightning_model: elm_lightning_model.Lightning_Model
    datamodule: elm_datamodule.ELM_Datamodule
    experiment_dir: str = './experiment_default'
    trial_name: str = None  # if None, use default Tensorboard scheme
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 50
    enable_progress_bar: bool = False
    log_freq: int = 100
    wandb_log: bool = False
    gradient_clip_value: int = None
    float_precision: str|int = 16 if torch.cuda.is_available() else 32

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
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
            if slurm_identifier:
                self.trial_name = f"run_{slurm_identifier}_{datetime_str}"
            else:
                self.trial_name = f"run_{datetime_str}"

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
        self.trial_dir.mkdir(parents=True, exist_ok=True)
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
        else:
            wandb_logger = None

        self.earlystopping_callback = EarlyStopping(
            monitor=self.lightning_model.monitor_metric,
            mode='min' if 'loss' in self.lightning_model.monitor_metric else 'max',
            min_delta=self.early_stopping_min_delta,
            patience=self.early_stopping_patience,
            log_rank_zero_only=True,
            verbose=True,
        )

        print("Model Summary:")
        print(ModelSummary(self.lightning_model, max_depth=-1))

    def run_all(
        self,
        restart_chpt_path: str = None,
        max_epochs: int = 2,
        skip_test: bool = False,
        skip_predict: bool = False,
    ):
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir

        # reset `best` metrics for early stopping and lr scheduler
        if self.earlystopping_callback:
            self.earlystopping_callback.best_score = self.earlystopping_callback.best_score * (
                torch.inf if 'loss' in self.lightning_model.monitor_metric 
                else -torch.inf
            )
            self.earlystopping_callback.wait_count = self.earlystopping_callback.wait_count * 0
        if self.lightning_model.lr_scheduler:
            self.lightning_model.lr_scheduler.best = self.lightning_model.lr_scheduler.best * (
                torch.inf if 'loss' in self.lightning_model.monitor_metric 
                else -torch.inf
            )
            self.lightning_model.lr_scheduler.num_bad_epochs = self.lightning_model.lr_scheduler.num_bad_epochs * 0

        # set callbacks
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor=self.lightning_model.monitor_metric,
                mode='min' if 'loss' in self.lightning_model.monitor_metric else 'max',
                save_last=True,
            ),
            self.earlystopping_callback,
        ]

        trainer = Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=self.loggers,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_freq,
            num_nodes=int(os.getenv('SLURM_NNODES', default=1)),
            precision=self.float_precision,
            strategy=DDPStrategy(find_unused_parameters=True)
        )
        self.datamodule.is_global_zero = trainer.is_global_zero

        trainer.fit(
            self.lightning_model, 
            datamodule=self.datamodule,
            ckpt_path=restart_chpt_path,
        )
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.last_model_path = trainer.checkpoint_callback.last_model_path
        
        if skip_test is False:
            trainer.test(datamodule=self.datamodule, ckpt_path='best')

        if skip_predict is False:
            trainer.predict(datamodule=self.datamodule, ckpt_path='best')

        print(f"Trial dir: {self.experiment_dir}")
        print(f"Trial name: {self.trial_name}")
        print(f"Trial dir: {self.trial_dir}")
        print(f"Best model path: {self.best_model_path}")
        print(f"Last model path: {self.last_model_path}")


if __name__=='__main__':
    signal_window_size = 256

    # ugly hack: must init Lightning model before Torch model
    lightning_model = elm_lightning_model.Lightning_Model()
    torch_model = elm_torch_model.Torch_CNN_Model(
        signal_window_size=signal_window_size,
        cnn_nlayers=6,
        cnn_num_kernels=4,
        cnn_kernel_time_size=2,
        cnn_padding=[[0,1,1]]*3 + [0]*3,
    )
    lightning_model.set_torch_model(torch_model=torch_model)

    datamodule = elm_datamodule.ELM_Datamodule(
        signal_window_size=signal_window_size,
        max_elms=5,
        batch_size=128,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        # wandb_log=True,
    )

    lightning_model.torch_model.frontends_active['time_to_elm_regression'] = False
    lightning_model.torch_model.frontends_active['active_elm_classification'] = False
    trainer.run_all(
        max_epochs=2,
        skip_predict=True,
    )
    lightning_model.torch_model.frontends_active['active_elm_classification'] = True
    trainer.run_all(
        max_epochs=2,
        skip_predict=True,
        restart_chpt_path=trainer.last_model_path,
    )
    lightning_model.torch_model.frontends_active['time_to_elm_regression'] = True
    trainer.run_all(
        max_epochs=2,
        restart_chpt_path=trainer.last_model_path,
    )
