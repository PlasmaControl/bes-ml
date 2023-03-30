import dataclasses
import time

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

import pyhessian

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb

@dataclasses.dataclass(eq=False)
class MNISTDataModule_DataClass():
    data_dir: str = './MNIST'
    batch_size: int = 32
    fraction_validation: float = 0.2
    seed: int = 42
    num_workers: int = 8


@dataclasses.dataclass(eq=False)
class MNISTDataModule(
    pl.LightningDataModule,
    MNISTDataModule_DataClass,
):
    def __post_init__(self):
        super().__init__()

    def setup(self, stage: str):
        transform = torchvision.transforms.ToTensor()
        mnist_full = torchvision.datasets.MNIST(
            self.data_dir, 
            download=True,
            train=True,
            transform=transform,
        )
        self.mnist_test = torchvision.datasets.MNIST(
            self.data_dir, 
            download=True,
            train=False,
            transform=transform,
        )

        # partition training and validation data
        valid_set_size = int(len(mnist_full) * self.fraction_validation)
        train_set_size = len(mnist_full) - valid_set_size
        seed = torch.Generator().manual_seed(self.seed)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_full, 
            [train_set_size, valid_set_size], 
            generator=seed,
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
    hidden_size_1: int = 128
    hidden_size_2: int = 32
    hessian_epoch_interval: int = 2
    dropout: float = 0.1
    leaky_relu_slope: float = 1e-2

@dataclasses.dataclass(eq=False)
class LitAutoEncoder(
    pl.LightningModule,
    LitAutoEncoder_DataClass,
):
    
    def __post_init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(4,28,28)
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
        self.loss_fn = F.cross_entropy

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
    
    def eval_and_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # def on_train_epoch_end(self) -> None:
    #     if self.hessian_epoch_interval and self.current_epoch%self.hessian_epoch_interval == 0:
    #         print('calc hess')
    #         t_start = time.time()
    #         hessian = pyhessian.hessian(
    #             model=self,
    #             criterion=self.loss_fn,
    #             dataloader=self.trainer.train_dataloader,
    #             cuda=torch.cuda.is_available(),
    #         )
    #         print('calc eigenvals')
    #         # eigenvalues, eigenvectors = hessian.eigenvalues(top_n=1, maxIter=5, tol=1e-2)
    #         # eigen_list, weight_list = hessian.density(iter=5)
    #         # print(eigen_list)
    #         # print(weight_list)
    #         # print(f"elapsed time {time.time()-t_start:.1f} s")
    #         eigen_vals, eigen_vecs = hessian.eigenvalues(maxIter=10, top_n=5)
    #         print(eigen_vals)
    #         eigen_list, weight_list = hessian.density(iter=10)
    #         print(eigen_list)
    #         print(weight_list)
    #         print('finished hess')
    #         print(f"elapsed time {time.time()-t_start:.1f} s")

    #         for param in self.parameters():
    #             param.grad = None


def dev_test():
    # define model
    autoencoder = LitAutoEncoder(
        hessian_epoch_interval=1,
    )
    datamodule = MNISTDataModule()

    # check model and data
    check_trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        devices=1,
    )
    check_trainer.fit(
        model=autoencoder, 
        datamodule=datamodule,
    )
    check_trainer.test(
        model=autoencoder,
        datamodule=datamodule
    )


def full_test():
    exp_dir = 'Experiment_test'

    autoencoder = LitAutoEncoder(
        lr=1e-4,
        hessian_epoch_interval=0,
    )
    datamodule = MNISTDataModule()

    tb_logger = TensorBoardLogger(
        save_dir='.', 
        name=exp_dir, 
        log_graph=True,
    )
    tb_logger.log_graph(autoencoder)

    wandb.login()
    wandb_logger = WandbLogger(
        save_dir=exp_dir,
        project=exp_dir,
        name=f"version_{tb_logger.version}"
    )
    wandb_logger.watch(autoencoder, log='all', log_freq=500)

    trainer = pl.Trainer(
        max_epochs=6,
        logger=[
            tb_logger,
            wandb_logger,
        ],
        # enable_progress_bar=False,
    )
    trainer.fit(
        model=autoencoder, 
        datamodule=datamodule,
    )
    trainer.test(
        model=autoencoder, 
        datamodule=datamodule,
    )

    wandb.finish(quiet=True)


if __name__=="__main__":
    dev_test()
    # full_test()