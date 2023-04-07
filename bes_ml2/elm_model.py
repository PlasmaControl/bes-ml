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
class Model_DataClass:
    signal_window_size: int = 128  # power of 2; ~16-512
    cnn_layer1_num_kernels: int = 16
    cnn_layer1_kernel_time_size: int = 5
    cnn_layer1_kernel_spatial_size: int = 3
    cnn_layer1_maxpool_time: int = 4
    cnn_layer2_num_kernels: int = 16
    cnn_layer2_kernel_time_size: int = 5
    cnn_layer2_kernel_spatial_size: int = 3
    cnn_layer2_maxpool_time: int = 4
    mlp_layer1_size: int = 64
    mlp_layer2_size: int = 32
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
        cnn_parameters = sum(p.numel() for p in self.featurize.parameters() if p.requires_grad)
        mlp_parameters = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"Total parameters {cnn_parameters+mlp_parameters:,}")
        print(f"  CNN parameters {cnn_parameters:,}")
        print(f"  MLP parameters {mlp_parameters:,}")

        # self.full_model = torch.nn.Sequential(
        #     self.featurize,
        #     self.mlp,
        # )

    def forward(self, x: torch.Tensor):
        features = self.featurize(x)
        prediction = self.mlp(features)
        return prediction


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
        self.r2_score = torchmetrics.R2Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx) -> torch.Tensor:
        x, y_target = batch
        y_prediction = self(x)
        loss = self.mse_loss(y_prediction, y_target)
        self.r2_score(y_prediction, y_target)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._shared_step(batch=batch, batch_idx=batch_idx)
        self.log("train_loss", self.mse_loss)
        # self.log("train_score", self.r2_score)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch=batch, batch_idx=batch_idx)
        self.log("val_loss", self.mse_loss)
        self.log("val_score", self.r2_score)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch=batch, batch_idx=batch_idx)
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
    
    # def on_predict_epoch_end(self, results) -> None:
    #     fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
    #     i_page = 1
    #     plt.suptitle(f"Inference on ELMs in test dataset (page {i_page})")
    #     axes_twinx = [axis.twinx() for axis in axes.flat]
    #     for i_elm, result in enumerate(results):
    #         labels = torch.concat([batch['labels'] for batch in result]).squeeze()
    #         predictions = torch.concat([batch['predictions'] for batch in result]).squeeze()
    #         assert labels.shape[0] == predictions.shape[0]
    #         dataloader = self.trainer.predict_dataloaders[i_elm]
    #         dataset: elm_data.ELM_Predict_Dataset = dataloader.dataset
    #         signal = dataset.signals[..., 2, 3].squeeze()
    #         assert signal.shape[0] == labels.shape[0]-1+self.signal_window_size
    #         time = (np.arange(signal.numel()) - dataset.active_elm_start_index)/1e3
    #         if i_elm % 6 == 0:
    #             for i_axis in range(axes.size):
    #                 axes.flat[i_axis].clear()
    #                 axes_twinx[i_axis].clear()
    #         plt.sca(axes.flat[i_elm%6])
    #         plt.plot(time[self.signal_window_size-1:], labels, label='Label')
    #         plt.plot(time[self.signal_window_size-1:], predictions, label='Prediction')
    #         plt.ylabel("Label | Prediction")
    #         plt.xlabel('Time to ELM (ms)')
    #         plt.legend(fontsize='small', loc='upper right')
    #         twinx = axes_twinx[i_elm%6]
    #         twinx.plot(time, signal, label='Signal', color='C2')
    #         twinx.set_ylabel('Scaled signal')
    #         twinx.legend(fontsize='small', loc='lower right')
    #         if i_elm % 6 == 5 or i_elm == len(results)-1:
    #             plt.tight_layout()
    #             filename = f'inference_{i_page:02d}'
    #             filepath = os.path.join(self.log_dir, filename)
    #             plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
    #             plt.savefig(filepath+'.png', format='png', transparent=True)
    #             for logger in self.loggers:
    #                 exp = logger.experiment
    #                 if isinstance(logger, loggers.TensorBoardLogger) and hasattr(exp, 'add_figure'):
    #                     exp.add_figure(filename, fig, close=False)
    #                 if isinstance(logger, loggers.WandbLogger) and hasattr(exp, 'add_image'):
    #                     exp.add_image(key='inference', images=[filepath+'.png'])
    #             i_page += 1
    #     plt.close(fig)


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
            'monitor': 'val_score',
        }
