import dataclasses
import os

import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.utils.data

from lightning.pytorch import LightningModule, loggers
import torchmetrics

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


@dataclasses.dataclass(eq=False)
class Lightning_Model(LightningModule):
    lr: float = 1e-3
    lr_scheduler_patience: int = 2
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-3
    monitor_metric: str = 'val_score'
    log_dir: str = dataclasses.field(default=None, init=False)
    
    def __post_init__(self):
        super().__init__()
        self.torch_model = None

    def set_torch_model(self, torch_model: torch.nn.Module = None):
        assert torch_model and hasattr(torch_model, 'signal_window_size')
        hp_fields = dataclasses.asdict(self) | dataclasses.asdict(torch_model)
        hp_fields['torch_model_name'] = torch_model.__class__.__name__
        for field in ['lr_scheduler_patience','lr_scheduler_threshold']:
            hp_fields.pop(field)
        self.save_hyperparameters(hp_fields)

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

        # initialize trainable parameters
        print("Initializing model layers")
        for name, param in self.torch_model.named_parameters():
            if name.endswith(".bias"):
                print(f"  {name}: initialized to zeros")
                param.data.fill_(0)
            else:
                dx = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / dx)
                print(f"  {name}: initialized to uniform +- {sqrt_k:.1e}")
                param.data.uniform_(-sqrt_k, sqrt_k)
                print(f"    dx*var: {dx*torch.var(param.data)}")
       
        sample_batch = torch.empty(
            (512, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )
        sample_batch.normal_()
        self.eval()
        with torch.no_grad():
            sample_batch_outputs = self(sample_batch)
        print(sample_batch_outputs.size())
        print(torch.mean(sample_batch_outputs))
        print(torch.var(sample_batch_outputs))

        self.example_input_array = torch.zeros(
            (2, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )
        self.mse_loss = torchmetrics.MeanSquaredError()
        self.r2_score = torchmetrics.R2Score()
        self.predict_outputs: list[list] = []

    def forward(self, signals):
        return self.torch_model(signals)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        signals, labels = batch
        predictions = self(signals)
        loss = self.mse_loss(predictions, labels)
        self.log("train_loss", self.mse_loss)
        return loss

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            for name, param in self.torch_model.named_parameters():
                if 'weight' in name:
                    values = param.data.detach()
                    mean = torch.mean(values).item()
                    std = torch.std(values).item()
                    z_scores = (values-mean)/std
                    skew = torch.mean(z_scores**3).item()
                    exkurt = torch.mean(z_scores**4).item() - 3
                    self.log(f"{name}.mean", mean, rank_zero_only=True)
                    self.log(f"{name}.std", std, rank_zero_only=True)
                    self.log(f"{name}.skew", skew, rank_zero_only=True)
                    self.log(f"{name}.exkurt", exkurt, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        predictions = self(signals)
        self.mse_loss(predictions, labels)
        self.log("val_loss", self.mse_loss)
        if self.current_epoch >= 25:
            self.r2_score(predictions, labels)
            self.log("val_score", self.r2_score)
        else:
            self.log("val_score", 0.)

    def test_step(self, batch, batch_idx):
        signals, labels = batch
        predictions = self(signals)
        self.mse_loss(predictions, labels)
        self.log("test_loss", self.mse_loss)
        self.r2_score(predictions, labels)
        self.log("test_score", self.r2_score)
        self.log("hp_metric", self.r2_score)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        signals, labels = batch
        predictions = self(signals)
        if batch_idx == 0:
            self.predict_outputs.append([])
            assert dataloader_idx == len(self.predict_outputs)-1
        self.predict_outputs[-1].append({
            'labels': labels,
            'predictions': predictions,
        })
    
    def on_predict_epoch_end(self) -> None:
        i_page = 1
        for i_elm, result in enumerate(self.predict_outputs):
            labels = torch.concat([batch['labels'] for batch in result]).squeeze()
            predictions = torch.concat([batch['predictions'] for batch in result]).squeeze()
            assert labels.shape[0] == predictions.shape[0]
            dataloader: torch.utils.data.DataLoader = self.trainer.predict_dataloaders[i_elm]
            dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
            signal = dataset.signals[..., 2, 3].squeeze()
            assert signal.shape[0] == labels.shape[0]-1+self.signal_window_size
            time = (np.arange(signal.numel()) - dataset.active_elm_start_index)/1e3
            if i_elm % 6 == 0:
                plt.close('all')
                fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
                plt.suptitle(f"Inference on ELMs in test dataset (page {i_page})")
            plt.sca(axes.flat[i_elm%6])
            plt.plot(time[self.signal_window_size-1:], labels, label='Label')
            plt.plot(time[self.signal_window_size-1:], predictions, label='Prediction')
            plt.ylabel("Label | Prediction")
            plt.xlabel('Time to ELM (ms)')
            plt.legend(fontsize='small', loc='upper right')
            twinx: matplotlib.axes.Axes = axes.flat[i_elm%6].twinx()
            twinx.plot(time, signal, label='Signal', color='C2')
            twinx.set_ylabel('Scaled signal')
            twinx.legend(fontsize='small', loc='lower right')
            if i_elm % 6 == 5 or i_elm == len(self.predict_outputs)-1:
                plt.tight_layout()
                filename = f'inference_{i_page:02d}'
                filepath = os.path.join(self.log_dir, filename)
                print(f"Saving figures {filepath}{{.pdf,.png}}")
                plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'.png', format='png', transparent=True)
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(filename, fig, close=False)
                    if isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='inference', images=[filepath+'.png'])
                i_page += 1
                plt.close(fig)
        self.predict_outputs.clear()

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
                min_lr=1e-5,
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
