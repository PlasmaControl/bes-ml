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
class BES_Trainer(
    elm_data.ELM_Datamodule_Dataclass,
    elm_model.Model_PL_DataClass,
):
    save_dir: str = '.'
    experiment_name: str = 'Experiment_test'
    max_epochs: int = 2
    gradient_clip_value: int = None
    version: str = None
    wandb_log: bool = False
    wandb_log_frequency: int = 20
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 10
    enable_progress_bar: bool = True

    def __post_init__(self):
        if self.wandb_log:
            wandb.login()
        self.datamodule = None
        self.model = None
        self.trainer = None
        self.create_datamodule()
        self.create_model()

    def run_fast_dev(self):
        tmp_trainer = pl.Trainer(
            fast_dev_run=True,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
        )
        tmp_trainer.fit(self.model, datamodule=self.datamodule)

    def create_datamodule(self):
        print('Creating data module')
        dm_class_fields_dict = {
            field.name: field 
            for field in dataclasses.fields(elm_data.ELM_Datamodule_Dataclass)
        }
        kwargs = {key: getattr(self, key) for key in dm_class_fields_dict}
        for key, value in kwargs.items():
            if value == dm_class_fields_dict[key].default:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}  (default {dm_class_fields_dict[key].default})")
        self.datamodule = elm_data.ELM_Datamodule(**kwargs)

    def create_model(self):
        print('Creating model')
        model_class_fields_dict = {
            field.name: field 
            for field in dataclasses.fields(elm_model.Model_PL_DataClass)
        }
        kwargs = {key: getattr(self, key) for key in model_class_fields_dict}
        for key, value in kwargs.items():
            if value == model_class_fields_dict[key].default:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}  (default {model_class_fields_dict[key].default})")
        self.model = elm_model.Model_PL(**kwargs)
        print("Model Summary:")
        print(ModelSummary(self.model, max_depth=-1))

    def make_loggers_and_callbacks(self):
        experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # set loggers
        self.pl_loggers = []
        tb_logger = loggers.TensorBoardLogger(
            save_dir=self.save_dir, 
            name=self.experiment_name, 
            version=self.version,
        )
        self.pl_loggers.append(tb_logger)

        version_str = tb_logger.version if tb_logger.version is str else f"version_{tb_logger.version}"

        if self.wandb_log:
            wandb_logger = loggers.WandbLogger(
                save_dir=experiment_dir,
                project=self.experiment_name,
                name=version_str,
            )
            wandb_logger.watch(
                self.model, 
                log='all', 
                log_freq=self.wandb_log_frequency,
            )
            self.pl_loggers.append(wandb_logger)

        # set callbacks
        self.callbacks = [
            cb.LearningRateMonitor(),
            cb.EarlyStopping(
                monitor='val_loss',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
            ),
        ]


    def initialize_trainer(self):
        print('Initiating trainer')
        inherited_fields = (
            [field.name for field in dataclasses.fields(elm_model.Model_PL_DataClass)] +
            [field.name for field in dataclasses.fields(elm_data.ELM_Datamodule_Dataclass)]
        )
        self_fields_dict = dataclasses.asdict(self)
        for field_name in inherited_fields:
            if field_name in self_fields_dict:
                self_fields_dict.pop(field_name)

        self_class_fields_dict = {
            field.name: field 
            for field in dataclasses.fields(self.__class__)
        }
        for field_name in self_fields_dict:
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = self_class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        assert self.model and self.datamodule
        assert os.path.exists(self.save_dir) is True

        self.make_loggers_and_callbacks()

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=self.pl_loggers,
            callbacks=self.callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=20,
        )

    def run_all(self):
        self.initialize_trainer()
        assert self.trainer and self.model and self.datamodule
        self.trainer.fit(self.model, datamodule=self.datamodule)
        self.trainer.test(datamodule=self.datamodule, ckpt_path='best')
        self.trainer.predict(datamodule=self.datamodule, ckpt_path='best')


if __name__=='__main__':
    trainer = BES_Trainer(
        max_epochs=10,
        batch_size=32,
        fraction_validation=0.1,
        fraction_test=0.1,
        # max_elms=5,
        wandb_log=True,
    )
    trainer.run_all()
