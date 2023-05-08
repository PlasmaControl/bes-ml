import dataclasses
import os
from datetime import datetime

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.utilities.model_summary import ModelSummary
import wandb

try:
    from . import elm_lightning_model
    from . import elm_torch_model
    from . import elm_datamodule
except:
    from bes_ml2 import elm_lightning_model
    from bes_ml2 import elm_torch_model
    from bes_ml2 import elm_datamodule



@dataclasses.dataclass(eq=False)
class BES_Trainer(
    elm_lightning_model.Lightning_Model_Dataclass,
    elm_torch_model.Torch_CNN_Model_Dataclass,
    elm_datamodule.ELM_Datamodule_Dataclass,
):
    experiment_group_dir: str = './experiment_default'
    experiment_name: str = None  # if None, use default Tensorboard scheme
    max_epochs: int = 2
    gradient_clip_value: int = None
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 10
    enable_progress_bar: bool = False
    skip_test_predict: bool = False
    lit_log_freq: int = 50
    wandb_log: bool = False
    wandb_log_freq: int = 100
    log_dir: str = dataclasses.field(default=None, init=False)

    def __post_init__(self):

        world_size = int(os.getenv("SLURM_NTASKS", 0))
        world_rank = int(os.getenv("SLURM_PROCID", 0))
        local_rank = int(os.getenv("SLURM_LOCALID", 0))
        node_rank = int(os.getenv("SLURM_NODEID", 0))
        print(f"World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})")

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        if self.experiment_name is None:
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_name = f"run_{datetime_str}"

        self.experiment_group_dir = os.path.abspath(self.experiment_group_dir)
        os.makedirs(self.experiment_group_dir, exist_ok=True)

        # set loggers
        loggers = []
        tb_logger = TensorBoardLogger(
            save_dir=os.path.dirname(self.experiment_group_dir), 
            name=os.path.basename(self.experiment_group_dir), 
            version=self.experiment_name,
            default_hp_metric=False,
        )
        loggers.append(tb_logger)
        os.makedirs(tb_logger.log_dir, exist_ok=True)
        self.log_dir = tb_logger.log_dir
        
        lit_model_fields = dataclasses.fields(elm_lightning_model.Lightning_Model_Dataclass)
        kwargs = {field.name: getattr(self, field.name) for field in lit_model_fields}
        self.lightning_model = elm_lightning_model.Lightning_Model(**kwargs)

        torch_model_fields = dataclasses.fields(elm_torch_model.Torch_CNN_Model_Dataclass)
        kwargs = {field.name: getattr(self, field.name) for field in torch_model_fields}
        torch_model = elm_torch_model.Torch_CNN_Model(**kwargs)

        self.lightning_model.set_torch_model(torch_model=torch_model)

        print("Model Summary:")
        print(ModelSummary(self.lightning_model, max_depth=-1))

        datamodule_fields = dataclasses.fields(elm_datamodule.ELM_Datamodule_Dataclass)
        kwargs = {field.name: getattr(self, field.name) for field in datamodule_fields}
        self.datamodule = elm_datamodule.ELM_Datamodule(**kwargs)

        if self.wandb_log:
            wandb.login()
            wandb_logger = WandbLogger(
                save_dir=self.experiment_group_dir,
                project=os.path.basename(self.experiment_group_dir),
                name=tb_logger.version,
            )
            wandb_logger.watch(
                self.lightning_model, 
                log='all', 
                log_freq=self.wandb_log_freq,
            )
            loggers.append(wandb_logger)

        # set callbacks
        callbacks = [
            LearningRateMonitor(),
            EarlyStopping(
                monitor=self.monitor_metric,
                mode='min' if 'loss' in self.monitor_metric else 'max',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
            ),
        ]

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=loggers,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.lit_log_freq,
            num_nodes=int(os.getenv('SLURM_NNODES', default=1)),
            devices="auto",
            accelerator="auto",
        )

    def run_all(self):
        self.trainer.fit(self.lightning_model, datamodule=self.datamodule)
        self.trainer.test(datamodule=self.datamodule)
        self.trainer.predict(datamodule=self.datamodule)


if __name__=='__main__':
    trainer = BES_Trainer(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        max_elms=10,
        signal_window_size=256,
        max_epochs=1,
        batch_size=128,
        cnn_nlayers=6,
        cnn_num_kernels=4,
        cnn_kernel_time_size=2,
        cnn_padding=[[0,1,1]]*3 + [0]*3,
        # fraction_validation=0.1,
        # fraction_test=0.1,
        enable_progress_bar=False,
        # wandb_log=True,
    )

    # signal_window_size = 256
    # world_size = int(os.getenv("SLURM_NTASKS", 0))
    # world_rank = int(os.getenv("SLURM_PROCID", 0))
    # local_rank = int(os.getenv("SLURM_LOCALID", 0))
    # node_rank = int(os.getenv("SLURM_NODEID", 0))
    # print(f"World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})")

    # if world_rank != 0:
    #     print(f"Sending world rank {world_rank} output to devnull")
    #     f = open(os.devnull, 'w')
    #     sys.stdout = f

    # try:
    #     """
    #     Step 1b: Initiate torch and lightning models
    #         Ugly hack: must initiate lightning model, then initiate torch model, 
    #         then add torch model to lightning model
    #     """
    #     lightning_model = elm_lightning_model.Lightning_Model()
    #     torch_model = elm_torch_model.Torch_CNN_Model(
    #         signal_window_size=signal_window_size,
    #         cnn_nlayers=6,
    #         cnn_num_kernels=4,
    #         cnn_kernel_time_size=2,
    #         cnn_padding=[[0,1,1]]*3 + [0]*3,
    #     )
    #     lightning_model.set_torch_model(torch_model=torch_model)

    #     """
    #     Step 1a: Initiate pytorch_lightning.LightningDataModule
    #     """
    #     datamodule = elm_datamodule.ELM_Datamodule(
    #         # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
    #         signal_window_size=signal_window_size,
    #         max_elms=10,
    #         batch_size=128,
    #         # fraction_validation=0.1,
    #         # fraction_test=0.1,
    #     )

    #     """
    #     Step 2: Initiate pytorch_lightning.Trainer and run
    #     """
    #     trainer = BES_Trainer(
    #         lightning_model=lightning_model,
    #         datamodule=datamodule,
    #         max_epochs=1,
    #         enable_progress_bar=False,
    #         # wandb_log=True,
    #     )
    #     trainer.run_all()
    # except:
    #     if world_rank != 0:
    #         f.close()
    #         sys.stdout = sys.__stdout__
    #     print(f"Error in world rank {world_rank}")
    #     raise
    # else:
    #     if world_rank != 0:
    #         f.close()
    #         sys.stdout = sys.__stdout__
