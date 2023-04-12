import dataclasses
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torchmetrics

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


@dataclasses.dataclass(eq=False)
class Lightning_Model(pl.LightningModule):
    lr: float = 1e-3
    lr_scheduler_patience: int = 2
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-3
    log_dir: str = '.'
    monitor_metric: str = 'val_score'
    
    def __post_init__(self):
        super().__init__()
        self.torch_model = None

    def set_torch_model(self, torch_model: torch.nn.Module = None):
        assert torch_model and hasattr(torch_model, 'signal_window_size')
        instance_fields = dataclasses.asdict(self) | dataclasses.asdict(torch_model)
        instance_fields['torch_model_name'] = torch_model.__class__.__name__
        self.save_hyperparameters(
            instance_fields,
            ignore=[
                'lr_scheduler_patience', 
                'lr_scheduler_threshold', 
            ],
        )

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        self.torch_model = torch_model
        self.signal_window_size = self.torch_model.signal_window_size
        self.example_input_array = torch.zeros(
            (2, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )
        self.mse_loss = torchmetrics.MeanSquaredError()
        self.r2_score = torchmetrics.R2Score()

    def forward(self, signals):
        return self.torch_model(signals)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        signals, labels = batch
        predictions = self(signals)
        loss = self.mse_loss(predictions, labels)
        self.log("train_loss", self.mse_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        predictions = self(signals)
        self.mse_loss(predictions, labels)
        self.log("val_loss", self.mse_loss)
        self.r2_score(predictions, labels)
        self.log("val_score", self.r2_score)

    def test_step(self, batch, batch_idx):
        signals, labels = batch
        predictions = self(signals)
        self.mse_loss(predictions, labels)
        self.log("test_loss", self.mse_loss)
        self.r2_score(predictions, labels)
        self.log("test_score", self.r2_score)
        self.log("hp_metric", self.r2_score)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> dict:
        signals, labels = batch
        predictions = self(signals)
        return {
            'labels': labels,
            'predictions': predictions,
        }
    
    def on_predict_epoch_end(self, results) -> None:
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
        i_page = 1
        plt.suptitle(f"Inference on ELMs in test dataset (page {i_page})")
        axes_twinx = [axis.twinx() for axis in axes.flat]
        for i_elm, result in enumerate(results):
            labels = torch.concat([batch['labels'] for batch in result]).squeeze()
            gather_labels = self.all_gather(labels)
            # labels = torch.concat()
            dataloader = self.trainer.predict_dataloaders[i_elm]
            dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
            print(f"{i_elm} {labels.shape} {gather_labels.shape} {dataset.signals.shape}")
        for i_elm, result in enumerate(results):
            labels = torch.concat([batch['labels'] for batch in result]).squeeze()
            predictions = torch.concat([batch['predictions'] for batch in result]).squeeze()
            assert labels.shape[0] == predictions.shape[0]
            dataloader = self.trainer.predict_dataloaders[i_elm]
            dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
            signal = dataset.signals[..., 2, 3].squeeze()
            if signal.shape[0] != labels.shape[0]-1+self.signal_window_size:
                print(i_elm)
                print(dataset.signals.shape)
                print(labels.shape[0]-1+self.signal_window_size)
                print(labels.shape)
            assert signal.shape[0] == labels.shape[0]-1+self.signal_window_size
            time = (np.arange(signal.numel()) - dataset.active_elm_start_index)/1e3
            if i_elm % 6 == 0:
                for i_axis in range(axes.size):
                    axes.flat[i_axis].clear()
                    axes_twinx[i_axis].clear()
            plt.sca(axes.flat[i_elm%6])
            plt.plot(time[self.signal_window_size-1:], labels, label='Label')
            plt.plot(time[self.signal_window_size-1:], predictions, label='Prediction')
            plt.ylabel("Label | Prediction")
            plt.xlabel('Time to ELM (ms)')
            plt.legend(fontsize='small', loc='upper right')
            twinx = axes_twinx[i_elm%6]
            twinx.plot(time, signal, label='Signal', color='C2')
            twinx.set_ylabel('Scaled signal')
            twinx.legend(fontsize='small', loc='lower right')
            if i_elm % 6 == 5 or i_elm == len(results)-1:
                plt.tight_layout()
                filename = f'inference_{i_page:02d}'
                filepath = os.path.join(self.log_dir, filename)
                plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'.png', format='png', transparent=True)
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(filename, fig, close=False)
                    if isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='inference', images=[filepath+'.png'])
                i_page += 1
        plt.close(fig)

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
                mode='min' if 'loss' in self.monitor_metric else 'max',
            ),
            'monitor': self.monitor_metric,
        }


@dataclasses.dataclass(eq=False)
class Lightning_Unsupervised_Model(Lightning_Model):
    monitor_metric: str = 'val_loss'

    def __post_init__(self):
        super().__post_init__()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        signals, _ = batch
        predictions = self(signals)
        loss = self.mse_loss(predictions, signals)
        self.log("train_loss", self.mse_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, _ = batch
        predictions = self(signals)
        self.mse_loss(predictions, signals)
        self.log("val_loss", self.mse_loss)

    def test_step(self, batch, batch_idx):
        signals, _ = batch
        predictions = self(signals)
        self.mse_loss(predictions, signals)
        self.log("test_loss", self.mse_loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> dict:
        pass
    
    def on_predict_epoch_end(self, results) -> None:
        pass
