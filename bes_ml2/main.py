import os
import sys
import dataclasses
from datetime import datetime
import time

from lightning.pytorch import loggers, callbacks, Trainer
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
    experiment_group_dir: str = './experiment_default'
    experiment_name: str = None  # if None, use default Tensorboard scheme
    max_epochs: int = 2
    gradient_clip_value: int = None
    wandb_log: bool = False
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 10
    enable_progress_bar: bool = False
    datamodule: elm_datamodule.ELM_Datamodule = None
    lightning_model: elm_lightning_model.Lightning_Model = None
    wandb_log_freq: int = 100
    lit_log_freq: int = 50
    skip_test_predict: bool = False
    is_global_zero: bool = True

    def __post_init__(self):
        assert self.datamodule and self.lightning_model
        assert self.datamodule.signal_window_size == self.lightning_model.signal_window_size

        self.monitor_metric = self.lightning_model.monitor_metric
        self.trainer = None

        if self.experiment_name is None:
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_name = f"run_{datetime_str}"

        if self.is_global_zero:
            print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            if field_name in ['datamodule', 'lightning_model']:
                continue
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            if self.is_global_zero:
                print(field_str)

        if self.is_global_zero:
            print("Model Summary:")
            print(ModelSummary(self.lightning_model, max_depth=-1))

    def make_loggers_and_callbacks(self):
        self.experiment_group_dir = os.path.abspath(self.experiment_group_dir)
        os.makedirs(self.experiment_group_dir, exist_ok=True)

        # set loggers
        self.loggers = []
        tb_logger = loggers.TensorBoardLogger(
            save_dir=os.path.dirname(self.experiment_group_dir), 
            name=os.path.basename(self.experiment_group_dir), 
            version=self.experiment_name,
            default_hp_metric=False,
        )
        self.loggers.append(tb_logger)
        os.makedirs(tb_logger.log_dir, exist_ok=True)
        self.lightning_model.log_dir = tb_logger.log_dir
        self.datamodule.log_dir = tb_logger.log_dir
        
        if self.wandb_log:
            wandb.login()
            wandb_logger = loggers.WandbLogger(
                save_dir=self.experiment_group_dir,
                project=os.path.basename(self.experiment_group_dir),
                name=tb_logger.version,
            )
            wandb_logger.watch(
                self.lightning_model, 
                log='all', 
                log_freq=self.wandb_log_freq,
            )
            self.loggers.append(wandb_logger)

        # set callbacks
        self.callbacks = [
            callbacks.LearningRateMonitor(),
            callbacks.EarlyStopping(
                monitor=self.monitor_metric,
                mode='min' if 'loss' in self.monitor_metric else 'max',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
            ),
        ]

    def run_all(self):
        self.make_loggers_and_callbacks()

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=self.loggers,
            callbacks=self.callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.lit_log_freq,
            num_nodes=int(os.getenv('SLURM_NNODES', default=1)),
            devices="auto",
            accelerator="auto",
        )
        assert self.trainer.is_global_zero == self.is_global_zero

        self.trainer.fit(self.lightning_model, datamodule=self.datamodule)
        
        if self.skip_test_predict:
            pass
        else:
            self.trainer.test(datamodule=self.datamodule, ckpt_path='best')
            self.trainer.predict(datamodule=self.datamodule, ckpt_path='best', return_predictions=True)


if __name__=='__main__':
    world_size = int(os.getenv("SLURM_NTASKS", 0))
    world_rank = int(os.getenv("SLURM_PROCID", 0))
    local_rank = int(os.getenv("SLURM_LOCALID", 0))
    node_rank = int(os.getenv("SLURM_NODEID", 0))
    print(f"World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})")

    is_global_zero = world_rank == 0

    if not is_global_zero:
        f = open(os.devnull, 'w')
        sys.stdout = f

    try:
        t_start = time.time()
        signal_window_size = 256

        lightning_model = elm_lightning_model.Lightning_Model(
            is_global_zero=is_global_zero,
        )
        assert lightning_model.global_rank == world_rank
        torch_model = elm_torch_model.Torch_CNN_Model(
            signal_window_size=signal_window_size,
            cnn_nlayers=6,
            cnn_num_kernels=4,
            cnn_kernel_time_size=2,
            cnn_padding=[[0,1,1]]*3 + [0]*3,
            is_global_zero=is_global_zero,
        )
        lightning_model.set_torch_model(torch_model=torch_model)

        datamodule = elm_datamodule.ELM_Datamodule(
            # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
            signal_window_size=signal_window_size,
            max_elms=5,
            batch_size=128,
            is_global_zero=is_global_zero,
        )

        trainer = BES_Trainer(
            lightning_model=lightning_model,
            datamodule=datamodule,
            max_epochs=2,
            # wandb_log=True,
            enable_progress_bar=False,
        )
        trainer.run_all()
        print(f"Elapsed time {(time.time()-t_start)/60:.1f} min")
    except:
        if not is_global_zero:
            f.close()
            sys.stdout = sys.__stdout__
        raise
    finally:
        if not is_global_zero:
            f.close()
            sys.stdout = sys.__stdout__
