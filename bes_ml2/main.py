import dataclasses
import os
from datetime import datetime

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
    enable_progress_bar: bool = True
    datamodule: elm_datamodule.ELM_Datamodule = None
    lightning_model: elm_lightning_model.Lightning_Model = None
    wandb_log_freq: int = 100
    lit_log_freq: int = 50
    skip_test_predict: bool = False

    def __post_init__(self):
        assert self.datamodule and self.lightning_model
        assert self.datamodule.signal_window_size == self.lightning_model.signal_window_size

        self.monitor_metric = self.lightning_model.monitor_metric
        self.trainer = None

        if self.experiment_name is None:
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_name = f"run_{datetime_str}"

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
            print(field_str)

        print("Model Summary:")
        print(ModelSummary(self.lightning_model, max_depth=-1))

    def run_fast_dev(self):
        tmp_trainer = Trainer(
            fast_dev_run=True,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
        )
        tmp_trainer.fit(
            model=self.lightning_model, 
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
            default_hp_metric=False,
        )
        self.loggers.append(tb_logger)
        # version_str = tb_logger.version if tb_logger.version is str else f"version_{tb_logger.version}"
        self.lightning_model.log_dir = tb_logger.log_dir
        
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
        self.trainer.fit(
            model=self.lightning_model, 
            datamodule=self.datamodule,
        )

        if self.skip_test_predict:
            return

        self.trainer.test(
            model=self.lightning_model,
            datamodule=self.datamodule, 
            ckpt_path='best',
        )
        self.trainer.predict(
            model=self.lightning_model,
            datamodule=self.datamodule, 
            ckpt_path='best',
        )

        # # ugly hack to properly predict a single ELM
        # if torch.distributed.is_initialized():
        #     torch.distributed.destroy_process_group()
        # if self.trainer.is_global_zero:
        #     tmp_trainer = pl.Trainer(
        #         enable_model_summary=False,
        #         enable_progress_bar=self.enable_progress_bar,
        #         num_nodes=1,
        #         num_processes=1,
        #         devices=1,
        #         accelerator="auto",
        #         logger=False,
        #     )
        #     tmp_trainer.predict(
        #         model=self.lightning_model, 
        #         datamodule=self.datamodule, 
        #         ckpt_path=self.trainer.checkpoint_callback.best_model_path,
        #     )


if __name__=='__main__':
    signal_window_size = 256

    """
    Step 1a: Initiate pytorch_lightning.LightningDataModule
    """
    datamodule = elm_datamodule.ELM_Datamodule(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        signal_window_size=signal_window_size,
        max_elms=100,
        batch_size=64,
        fraction_validation=0.1,
        fraction_test=0.1,
    )

    """
    Step 1b: Initiate torch and lightning models
        Ugly hack: must initiate lightning model, then initiate torch model, 
        then add torch model to lightning model
    """
    lightning_model = elm_lightning_model.Lightning_Model()
    torch_model = elm_torch_model.Torch_CNN_Model(
        signal_window_size=signal_window_size,
        cnn_nlayers=6,
        cnn_num_kernels=4,
        cnn_kernel_time_size=2,
        cnn_padding=[[0,1,1]]*3 + [0]*3,
    )
    # lightning_model = elm_lightning_model.Lightning_Unsupervised_Model()
    # torch_model = elm_torch_model.Torch_AE_Model(
    #     signal_window_size=signal_window_size,
    #     cnn_nlayers=6,
    #     cnn_num_kernels=4,
    #     cnn_kernel_time_size=2,
    #     cnn_padding=[[0,1,1]]*3 + [0]*3,
    # )
    lightning_model.set_torch_model(torch_model=torch_model)

    """
    Step 2: Initiate pytorch_lightning.Trainer and run
    """
    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        max_epochs=2,
        # wandb_log=True,
        # skip_test_predict=True,
    )
    trainer.run_all()
