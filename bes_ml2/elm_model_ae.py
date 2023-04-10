import dataclasses
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torchmetrics

import elm_data


@dataclasses.dataclass(eq=False)
class Model_DataClass:
    signal_window_size: int = 128  # power of 2; ~16-512
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
    leaky_relu_slope: float = 1e-2
    dropout: float = 0.1


@dataclasses.dataclass(eq=False)
class Model(
    torch.nn.Module,
    Model_DataClass,
):
    
    def __post_init__(self):
        super().__init__()
        print('Constructing CNN')
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
        self.data_shape_after_maxpool1 = data_shape.copy()

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
        self.data_shape_after_maxpool2 = data_shape.copy()
        # CNN output features
        cnn_features = np.prod(data_shape)
        print(f"  CNN output features {cnn_features}")

        # CNN model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer1_kernel, return_indices=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=self.cnn_layer2_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer2_kernel, return_indices=True),
        )
        self.max_unpool2 = torch.nn.MaxUnpool3d(kernel_size=maxpool_layer2_kernel)
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                in_channels=self.cnn_layer2_num_kernels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
        )
        self.max_unpool1 = torch.nn.MaxUnpool3d(kernel_size=maxpool_layer1_kernel)
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=in_channels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
        )

        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")

    def forward(self, signals: torch.Tensor):
        out, indices1 = self.conv1(signals)
        out, indices2 = self.conv2(out)
        out = self.max_unpool2(out, indices2)
        out = self.deconv2(out)
        out = self.max_unpool1(out, indices1)
        reconstructed_signals = self.deconv1(out)
        return reconstructed_signals


@dataclasses.dataclass(eq=False)
class Model_PL_DataClass(Model_DataClass):
    lr: float = 1e-3
    lr_scheduler_patience: int = 2
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-3
    gradient_clip_value: int = None  # added here for save_hyperparameters()
    log_dir: str = '.'

@dataclasses.dataclass(eq=False)
class Model_PL(
    pl.LightningModule,
    Model_PL_DataClass,
):
    
    def __post_init__(self):
        super().__init__()
        self.example_input_array = torch.zeros((2, 1, self.signal_window_size, 8, 8), dtype=torch.float32)
        self.save_hyperparameters(ignore=['lr_scheduler_patience', 'lr_scheduler_threshold'])
        model_class_fields_dict = {field.name: field for field in dataclasses.fields(Model_DataClass)}
        model_kwargs = {key: getattr(self, key) for key in model_class_fields_dict}
        self.model = Model(**model_kwargs)
        self.mse_loss = torchmetrics.MeanSquaredError()
        self.monitor_metric = 'val_loss'

    def forward(self, signals):
        return self.model(signals)

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
        signals, _ = batch
        predictions = self(signals)
        return {
            'signals': signals,
            'predictions': predictions,
        }
    
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
                mode='min' if self.monitor_metric.endswith('loss') else 'max',
            ),
            'monitor': self.monitor_metric,
        }
