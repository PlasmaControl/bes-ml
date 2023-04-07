import dataclasses
import os

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import callbacks as cb
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.core.mixins import HyperparametersMixin

import wandb

@dataclasses.dataclass(eq=False)
class MNISTDataModule_DataClass():
    data_dir: str = './MNIST'
    batch_size: int = 32
    fraction_validation: float = 0.2
    num_workers: int = 4
    seed: int = 42


@dataclasses.dataclass(eq=False)
class MNISTDataModule(
    pl.LightningDataModule,
    MNISTDataModule_DataClass,
):
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters('batch_size')

    def prepare_data(self):
        for train in [True, False]:
            torchvision.datasets.MNIST(
                self.data_dir, 
                download=True,
                train=train,
            )

    def setup(self, stage: str):
        # train and validation data
        transform = torchvision.transforms.ToTensor()
        mnist_full = torchvision.datasets.MNIST(
            self.data_dir, 
            train=True,
            transform=transform,
        )
        # partition
        valid_set_size = int(len(mnist_full) * self.fraction_validation)
        train_set_size = len(mnist_full) - valid_set_size
        seed = torch.Generator().manual_seed(self.seed)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_full, 
            [train_set_size, valid_set_size], 
            generator=seed,
        )

        # test data
        self.mnist_test = torchvision.datasets.MNIST(
            self.data_dir, 
            train=False,
            transform=transform,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

@dataclasses.dataclass(eq=False)
class LitAutoEncoder_DataClass():
    hidden_size_1: int = 128
    hidden_size_2: int = 32
    dropout: float = 0.1
    leaky_relu_slope: float = 1e-2
    lr: float = 1e-3
    lr_scheduler_patience: int = 2
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-3
    gradient_clip_value: int = None  # added here for save_hyperparameters()

@dataclasses.dataclass(eq=False)
class LitAutoEncoder(
    pl.LightningModule,
    LitAutoEncoder_DataClass,
):
    
    def __post_init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(2, 28, 28)
        self.save_hyperparameters(ignore=['lr_scheduler_patience', 'lr_scheduler_threshold'])
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, self.hidden_size_1),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size_1, self.hidden_size_2),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size_2, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
    
    def eval_and_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.eval_and_loss(batch)
        self.log("tr_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.eval_and_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.eval_and_loss(batch)
        self.log("hp_metric", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.5,
                patience=self.lr_scheduler_patience,
                threshold=self.lr_scheduler_threshold,
                min_lr=1e-6,
            ),
            'monitor': 'val_loss',
        }


@dataclasses.dataclass(eq=False)
class BES_Trainer(
    LitAutoEncoder_DataClass,
    MNISTDataModule_DataClass,
):
    save_dir: str = '.'
    experiment_name: str = 'Experiment_test'
    max_epochs: int = 4
    gradient_clip_value: int = None
    version: str = None
    wandb_log: bool = True
    wandb_log_frequency: int = 500
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

        # fast dev run
        tmp_trainer = pl.Trainer(
            fast_dev_run=True,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
        )
        tmp_trainer.fit(self.model, datamodule=self.datamodule)

        # self.initialize_trainer()

    def create_datamodule(self):
        print('Creating data module')
        dm_class_fields_dict = {field.name: field for field in dataclasses.fields(MNISTDataModule_DataClass)}
        kwargs = {key: getattr(self, key) for key in dm_class_fields_dict}
        for key, value in kwargs.items():
            if value == dm_class_fields_dict[key].default:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}  (default {dm_class_fields_dict[key].default})")
        self.datamodule = MNISTDataModule(**kwargs)

    def create_model(self):
        print('Creating model')
        model_class_fields_dict = {field.name: field for field in dataclasses.fields(LitAutoEncoder_DataClass)}
        kwargs = {key: getattr(self, key) for key in model_class_fields_dict}
        for key, value in kwargs.items():
            if value == model_class_fields_dict[key].default:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}  (default {model_class_fields_dict[key].default})")
        self.model = LitAutoEncoder(**kwargs)
        print(ModelSummary(self.model, max_depth=-1))

    def initialize_trainer(self):
        print('Initiating trainer')
        dm_class_fields_dict = {field.name: field for field in dataclasses.fields(MNISTDataModule_DataClass)}
        model_class_fields_dict = {field.name: field for field in dataclasses.fields(LitAutoEncoder_DataClass)}
        self_class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        self_fields_dict = dataclasses.asdict(self)
        for key in (dm_class_fields_dict | model_class_fields_dict):
            if key in self_fields_dict:
                self_fields_dict.pop(key)
        for key in self_fields_dict:
            value = getattr(self, key)
            if value == self_class_fields_dict[key].default:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}  (default {self_class_fields_dict[key].default})")

        assert self.model and self.datamodule
        assert os.path.exists(self.save_dir) is True

        experiment_dir = os.path.join(self.save_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # set loggers
        loggers = []
        tb_logger = TensorBoardLogger(
            save_dir=self.save_dir, 
            name=self.experiment_name, 
            version=self.version,
            # log_graph=True,
        )
        # tb_logger.log_graph(self.model)
        loggers.append(tb_logger)

        version_str = tb_logger.version if tb_logger.version is str else f"version_{tb_logger.version}"

        if self.wandb_log:
            wandb_logger = WandbLogger(
                save_dir=experiment_dir,
                project=self.experiment_name,
                name=version_str,
            )
            wandb_logger.watch(self.model, log='all', log_freq=self.wandb_log_frequency)
            loggers.append(wandb_logger)

        # set callbacks
        callbacks = [
            cb.LearningRateMonitor(),
            cb.EarlyStopping(
                monitor='val_loss',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
            ),
        ]

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_value,
            logger=loggers,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=self.enable_progress_bar,
        )

    def fit_and_test(self):
        assert self.trainer and self.model and self.datamodule
        # add any addition hparams
        print('Running fit()')
        self.trainer.fit(self.model, datamodule=self.datamodule)
        print('Running test()')
        self.trainer.test(self.model, datamodule=self.datamodule)
        if self.wandb_log:
            wandb.finish()


if __name__=="__main__":
    tr = BES_Trainer(
        experiment_name='Experiment_10',
        max_epochs=4,
        # wandb_log=False,
    )
    lr_vals = [1e-3]
    wd_vals = [1e-4]
    grad_clip_vals = [None]
    batch_size_vals = [128]
    for batch_size in batch_size_vals:
        tr.batch_size = batch_size
        tr.create_datamodule()
        for lr in lr_vals:
            for wd in wd_vals:
                for grad_clip in grad_clip_vals:
                        tr.lr = lr
                        tr.weight_decay = wd
                        tr.gradient_clip_value = grad_clip
                        tr.create_model()
                        tr.initialize_trainer()
                        tr.fit_and_test()
