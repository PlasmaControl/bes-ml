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

import elm_data


@dataclasses.dataclass(eq=False)
class Model_Base_Dataclass:
    signal_window_size: int = 128  # power of 2; ~16-512
    leaky_relu_slope: float = 1e-2
    dropout: float = 0.1
    mlp_layers_size: tuple = (64, 32)


@dataclasses.dataclass(eq=False)
class Torch_Model_CNN01(
    torch.nn.Module,
    Model_Base_Dataclass,
):
    cnn_layer1_num_kernels: int = 8
    cnn_layer1_kernel_time_size: int = 5
    cnn_layer1_kernel_spatial_size: int = 3
    cnn_layer1_maxpool_time: int = 4
    cnn_layer2_num_kernels: int = 8
    cnn_layer2_kernel_time_size: int = 5
    cnn_layer2_kernel_spatial_size: int = 3
    cnn_layer2_maxpool_time: int = 4
    mlp_layer1_size: int = 32
    mlp_layer2_size: int = 16
    
    def __post_init__(self):
        super().__init__()

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'
        assert self.cnn_layer1_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer2_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer1_maxpool_time%2 == 0
        assert self.cnn_layer2_maxpool_time%2 == 0
        in_channels = 1
        data_shape = [in_channels, self.signal_window_size, 8, 8]
        print(f"  Input data shape {data_shape}")

        # CNN and maxpool 1
        cnn_layer1_kernel = (
            self.cnn_layer1_kernel_time_size,
            self.cnn_layer1_kernel_spatial_size,
            self.cnn_layer1_kernel_spatial_size,
        )
        print(f"  CNN 1 kernel shape {cnn_layer1_kernel}")
        print(f"  CNN 1 kernel number {self.cnn_layer1_num_kernels}")
        cnn_layer1_padding = ((self.cnn_layer1_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer1_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer1_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer1_kernel_spatial_size-1)
        print(f"    Data shape after CNN 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer1_kernel = (self.cnn_layer1_maxpool_time, 1, 1)
        print(f"  Maxpool 1 shape {maxpool_layer1_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer1_maxpool_time
        print(f"    Data shape after maxpool 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"

        # CNN and maxpool 2
        cnn_layer2_kernel = (
            self.cnn_layer2_kernel_time_size,
            self.cnn_layer2_kernel_spatial_size,
            self.cnn_layer2_kernel_spatial_size,
        )
        print(f"  CNN 2 kernel shape {cnn_layer2_kernel}")
        print(f"  CNN 2 kernel number {self.cnn_layer2_num_kernels}")
        cnn_layer2_padding = ((self.cnn_layer2_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer2_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer2_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer2_kernel_spatial_size-1)
        print(f"    Data shape after CNN 2 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer2_kernel = (self.cnn_layer2_maxpool_time, 1, 1)
        print(f"  Maxpool 2 shape {maxpool_layer2_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer2_maxpool_time
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        print(f'    Data shape after maxpool 2 {data_shape}')

        # CNN output features
        cnn_features = np.prod(data_shape)
        print(f"  CNN output features {cnn_features}")

        # CNN model
        self.featurize = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer1_kernel),
            torch.nn.Conv3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=self.cnn_layer2_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer2_kernel),
        )

        # MLP
        print("Constructing MLP")
        print(f"  MLP layer 1 size {self.mlp_layer1_size}")
        print(f"  MLP layer 2 size {self.mlp_layer2_size}")
        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=cnn_features,
                out_features=self.mlp_layer1_size,
            ),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(
                in_features=self.mlp_layer1_size, 
                out_features=self.mlp_layer2_size,
            ),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(
                in_features=self.mlp_layer2_size, 
                out_features=1,
            ),
        )

        # parameter count
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.featurize.parameters() if p.requires_grad)
        mlp_parameters = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"  CNN parameters {cnn_parameters:,}")
        print(f"  MLP parameters {mlp_parameters:,}")

    def forward(self, signals: torch.Tensor):
        features = self.featurize(signals)
        prediction = self.mlp(features)
        return prediction


@dataclasses.dataclass(eq=False)
class Torch_Model_CNN02(
    torch.nn.Module,
    Model_Base_Dataclass,
):
    cnn_num_kernels: tuple = (16,16,16)
    cnn_kernel_time_size: tuple = (8,4,4)
    cnn_kernel_spatial_size: tuple = (3, 3, 3)
    
    def __post_init__(self):
        super().__init__()

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'
        nlayers = len(self.cnn_num_kernels)
        assert nlayers == len(self.cnn_kernel_spatial_size)
        assert nlayers == len(self.cnn_kernel_time_size)
        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        print("Constructing CNN layers")

        in_channels = 1
        data_shape = [in_channels, self.signal_window_size, 8, 8]
        print(f"  Input data shape {data_shape}")

        self.cnn_layers = torch.nn.Sequential()

        # CNN layers
        for i in range(nlayers):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[i],
            )
            stride = (self.cnn_kernel_time_size[i], 1, 1)
            conv3d = torch.nn.Conv3d(
                in_channels=in_channels if i==0 else self.cnn_num_kernels[i-1],
                out_channels=self.cnn_num_kernels[i],
                kernel_size=kernel,
                stride=stride,
            )
            data_shape = conv3d(torch.zeros(size=data_shape)).size()
            print(f"  Data shape after CNN layer {i}: {data_shape}")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after CNN layer {i}"
            self.cnn_layers.extend([
                conv3d,
                torch.nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            ])

        cnn_features = np.prod(data_shape)
        print(f"  CNN output features: {cnn_features}")
            
        # MLP layers
        self.mlp_layers = torch.nn.Sequential(torch.nn.Flatten())
        for i, layer_size in enumerate(self.mlp_layers_size):
            self.mlp_layers.extend([
                torch.nn.Linear(
                    in_features=cnn_features if i==0 else self.mlp_layers_size[i-1],
                    out_features=layer_size,
                ),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            ])

        # output node
        self.mlp_layers.append(
            torch.nn.Linear(in_features=self.mlp_layers_size[-1], out_features=1)
        )

        # parameter count
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn_layers.parameters() if p.requires_grad)
        mlp_parameters = sum(p.numel() for p in self.mlp_layers.parameters() if p.requires_grad)
        print(f"  CNN parameters {cnn_parameters:,}")
        print(f"  MLP parameters {mlp_parameters:,}")

    def forward(self, signals: torch.Tensor):
        features = self.cnn_layers(signals)
        prediction = self.mlp_layers(features)
        return prediction


@dataclasses.dataclass(eq=False)
class Lit_Model(pl.LightningModule):
    lr: float = 1e-3
    lr_scheduler_patience: int = 2
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-3
    log_dir: str = '.'
    monitor_metric: str = 'val_score'
    
    def __post_init__(self):
        super().__init__()
        self.torch_model = None

    def set_torch_model(self, torch_model: Model_Base_Dataclass = None):
        assert torch_model
        self.torch_model = torch_model
        dataclasses.asdict(self)
        dataclasses.asdict(torch_model)
        instance_fields = dataclasses.asdict(self) | dataclasses.asdict(torch_model)
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
        self.r2_score(predictions, labels)
        self.log("val_loss", self.mse_loss)
        self.log("val_score", self.r2_score)

    def test_step(self, batch, batch_idx):
        signals, labels = batch
        predictions = self(signals)
        self.mse_loss(predictions, labels)
        self.r2_score(predictions, labels)
        self.log("test_loss", self.mse_loss)
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
            predictions = torch.concat([batch['predictions'] for batch in result]).squeeze()
            assert labels.shape[0] == predictions.shape[0]
            dataloader = self.trainer.predict_dataloaders[i_elm]
            dataset: elm_data.ELM_Predict_Dataset = dataloader.dataset
            signal = dataset.signals[..., 2, 3].squeeze()
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
