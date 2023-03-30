import dataclasses

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

import wandb

@dataclasses.dataclass(eq=False)
class MNISTDataModule_DataClass():
    data_dir: str = './MNIST'
    batch_size: int = 32
    fraction_validation: float = 0.2
    num_workers: int = 8
    seed: int = 42


@dataclasses.dataclass(eq=False)
class MNISTDataModule(
    pl.LightningDataModule,
    MNISTDataModule_DataClass,
):
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

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
    lr: float = 1e-3
    weight_decay: float = 1e-3
    hidden_size_1: int = 128
    hidden_size_2: int = 32
    dropout: float = 0.1
    leaky_relu_slope: float = 1e-2
    scheduler_patience: int = 2
    scheduler_threshold: float = 1e-3

@dataclasses.dataclass(eq=False)
class LitAutoEncoder(
    pl.LightningModule,
    LitAutoEncoder_DataClass,
):
    
    def __post_init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(1, 28,28)
        self.save_hyperparameters()
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
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.5,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                min_lr=1e-5,
            ),
            'monitor': 'val_loss',
        }


def dev_test():
    # define model
    model = LitAutoEncoder()
    print(ModelSummary(model, max_depth=-1))
    dm = MNISTDataModule(num_workers=1)

    # check model and data
    check_trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    check_trainer.fit(model, dm)
    check_trainer.test(model, dm)


def full_test():
    exp_dir = 'Experiment_test'

    model = LitAutoEncoder(scheduler_threshold=5e-2, dropout=0.2)
    print(ModelSummary(model, max_depth=-1))
    dm = MNISTDataModule()

    # set loggers
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir='.', 
        name=exp_dir, 
        log_graph=True,
    )
    tb_logger.log_graph(model)
    loggers.append(tb_logger)

    wandb.login()
    wandb_logger = WandbLogger(
        save_dir=exp_dir,
        project=exp_dir,
        name=f"version_{tb_logger.version}"
    )
    wandb_logger.watch(model, log='all', log_freq=500)
    loggers.append(wandb_logger)

    # set callbacks
    callbacks = [
        cb.LearningRateMonitor(),
        cb.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=3,
        ),
    ]

    # do train
    trainer = pl.Trainer(
        max_epochs=12,
        gradient_clip_val=0.05,
        enable_model_summary=False,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    wandb.finish(quiet=True)


if __name__=="__main__":
    # dev_test()
    full_test()