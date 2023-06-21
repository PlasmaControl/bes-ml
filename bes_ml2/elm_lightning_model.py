from __future__ import annotations
import dataclasses
import os
import time
from typing import Iterable, Callable, Mapping

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn
from lightning.pytorch import LightningModule, loggers


@dataclasses.dataclass(eq=False)
class Torch_Base(LightningModule):
    signal_window_size: int = 128  # power of 2; ~64-512
    leaky_relu_slope: float = 1e-2

    def __post_init__(self):
        super().__init__()
        assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'


    def print_fields(self):
        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

    def initialize_layers(self):
        # initialize trainable parameters
        print("Initializing model layers")
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                print(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith(".weight"):
                n_in = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / n_in)
                print(f"  {name}: initialized to uniform +- {sqrt_k:.1e} (numel {param.data.numel()})")
                param.data.uniform_(-sqrt_k, sqrt_k)
                print(f"    n_in*var: {n_in*torch.var(param.data):.3f}")


class LitWrapper(LightningModule):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model
    
    def forward(self, inputs):
        return self.torch_model(inputs)


@dataclasses.dataclass(eq=False)
class Torch_MLP_Mixin(Torch_Base):
    mlp_layers: tuple = (64, 32)
    mlp_dropout: float = 0.1
    mlp_batchnorm: bool = True

    def make_mlp(
            self, 
            mlp_in_features: int, 
            mlp_out_features: int = 1,
            with_sigmoid: bool = False,
    ) -> torch.nn.Module:

        if self.mlp_batchnorm:
            print(f"  Batchnorm is active, so setting mlp_dropout=0")
            self.mlp_dropout = 0.0

        # MLP layers
        print("Constructing MLP layers")
        mlp_layers = torch.nn.Sequential(torch.nn.Flatten())
        n_layers = len(self.mlp_layers)
        for i, layer_size in enumerate(self.mlp_layers):
            in_features = mlp_in_features if i==0 else self.mlp_layers[i-1]
            print(f"  MLP layer {i} with in/out features: {in_features}/{layer_size} (LeakyReLU activ.)")
            if self.mlp_batchnorm:
                mlp_layers.append(torch.nn.BatchNorm1d(num_features=in_features))
            elif self.mlp_dropout and i != n_layers-1:
                mlp_layers.append(torch.nn.Dropout(p=self.mlp_dropout))
            mlp_layers.append(torch.nn.Linear(
                in_features=in_features,
                out_features=layer_size,
            ))
            mlp_layers.append(torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope))

        # output layer
        print(f"  MLP output layer with in/out features {self.mlp_layers[-1]}/{mlp_out_features} (no activ.)")
        mlp_layers.append(
            torch.nn.Linear(
                in_features=self.mlp_layers[-1], 
                out_features=mlp_out_features,
            )
        )

        # Logit or probability output?
        if with_sigmoid:
            print(f"  Applying sigmoid at MLP output for probability with range [0,1]")
            mlp_layers.append(torch.nn.Sigmoid())
        else:
            print(f"  Logit output (log odds, log(p/(1-p))) with range [-inf,inf]; use sigmoid to get prob.")

        return LitWrapper(mlp_layers)


@dataclasses.dataclass(eq=False)
class Torch_CNN_Mixin(Torch_Base):
    cnn_nlayers: int = 3
    cnn_num_kernels: Iterable|int = 16
    cnn_kernel_time_size: Iterable|int = 4
    cnn_kernel_spatial_size: Iterable|int = 3
    cnn_padding: Iterable|int|str = 0
    cnn_input_channels: int = 1
    cnn_dropout: float = 0.1
    cnn_batchnorm: bool = True

    def make_cnn_encoder(self) -> tuple[torch.nn.Module,int,tuple]:
        for attr_name in [
            'cnn_num_kernels',
            'cnn_kernel_time_size',
            'cnn_kernel_spatial_size',
            'cnn_padding',
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.cnn_nlayers, f"{attr_name} {attr_value}"
            else:
                new_attr_value = tuple([attr_value]*self.cnn_nlayers)
                setattr(self, attr_name, new_attr_value)

        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        if self.cnn_batchnorm:
            print(f"  Batchnorm is active, so setting cnn_dropout=0")
            self.cnn_dropout = 0.0

        print("Constructing CNN layers")

        data_shape = (self.cnn_input_channels, self.signal_window_size, 8, 8)
        self.input_data_shape = tuple(data_shape)
        print(f"  Input data shape {data_shape}  (size {np.prod(data_shape)})")

        # CNN layers
        cnn = torch.nn.Sequential()
        for i in range(self.cnn_nlayers):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[i],
            )
            stride = (self.cnn_kernel_time_size[i], 1, 1)
            print(f"  CNN Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {self.cnn_padding[i]}")
            conv3d = torch.nn.Conv3d(
                in_channels=self.cnn_num_kernels[i-1] if i>0 else self.cnn_input_channels,
                out_channels=self.cnn_num_kernels[i],
                kernel_size=kernel,
                stride=stride,
                padding=self.cnn_padding[i],
                padding_mode='reflect',
            )
            data_shape = tuple(conv3d(torch.zeros(size=data_shape)).size())
            print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after CNN layer {i}"
            if i>0:
                if self.cnn_batchnorm:
                    cnn.append(torch.nn.BatchNorm3d(num_features=self.cnn_num_kernels[i-1]))
                elif self.cnn_dropout:
                    cnn.append(torch.nn.Dropout(p=self.cnn_dropout))
            cnn.append(conv3d)
            cnn.append(torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope))

        num_features = np.prod(data_shape)
        print(f"  CNN output features: {num_features}")

        return LitWrapper(cnn), num_features, data_shape

    def make_cnn_decoder(self, input_data_shape: Iterable) -> torch.nn.Module:
        decoder = torch.nn.Sequential()
        data_shape = input_data_shape
        for i in range(self.cnn_nlayers-1, -1, -1):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[i],
            )
            stride = (self.cnn_kernel_time_size[i], 1, 1)
            print(f"  Decoder Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {self.cnn_padding[i]}")
            conv3d = torch.nn.ConvTranspose3d(
                in_channels=self.cnn_num_kernels[i],
                out_channels=self.cnn_num_kernels[i-1] if i>0 else self.cnn_input_channels,
                kernel_size=kernel,
                stride=stride,
                padding=self.cnn_padding[i],
            )
            data_shape = tuple(conv3d(torch.zeros(size=data_shape)).size())
            print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after Decoder layer {i}"
            if self.cnn_batchnorm and i>0:
                decoder.append(torch.nn.BatchNorm3d(num_features=self.cnn_num_kernels[i]))
            decoder.append(conv3d)
            if i > 0:
                decoder.append(torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope))
    
        assert np.array_equal(self.input_data_shape, data_shape)
    
        return LitWrapper(decoder)


@dataclasses.dataclass(eq=False)
class Lightning_Model(
    Torch_CNN_Mixin,
    Torch_MLP_Mixin,
):
    lr: float = 1e-3
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    monitor_metric: str = 'sum_loss/val'
    log_dir: str = dataclasses.field(default='.', init=False)
    # the following must be listed in `_frontend_names`
    reconstruction_decoder: bool = True
    time_to_elm_mlp: bool = True
    classifier_25_mlp: bool = True
    classifier_50_mlp: bool = True
    classifier_75_mlp: bool = True
    _frontend_names = ['reconstruction_decoder', 'time_to_elm_mlp', 
                       'classifier_25_mlp', 'classifier_50_mlp', 'classifier_75_mlp']
    
    def __post_init__(self):
        super().__post_init__()
        self.save_hyperparameters()

        self.print_fields()

        # CNN encoder `backend` to featurize input data
        self.cnn_encoder, cnn_features, cnn_output_shape = self.make_cnn_encoder()

        # `frontends` for regression, classification, and self-supervised learning
        self.frontends: Mapping[str, LightningModule] = torch.nn.ModuleDict()
        self.losses_and_scores: Mapping[str, Callable] = {}
        self.unfreeze_epoch: Mapping[str, int] = {}
        
        for frontend_key in self._frontend_names:
            assert hasattr(self, frontend_key)
            if not getattr(self, frontend_key):
                continue
            self.unfreeze_epoch[frontend_key] = 0
            if frontend_key == 'reconstruction_decoder':
                self.frontends[frontend_key] = self.make_cnn_decoder(input_data_shape=cnn_output_shape)
                self.losses_and_scores[f"{frontend_key}_mse_loss"] = torch.nn.functional.mse_loss
            elif frontend_key == 'time_to_elm_mlp':
                self.frontends[frontend_key] = self.make_mlp(mlp_in_features=cnn_features)
                self.losses_and_scores[f"{frontend_key}_mse_loss"] = torch.nn.functional.mse_loss
                self.losses_and_scores[f"{frontend_key}_r2_score"] = sklearn.metrics.r2_score
            elif 'classifier' in frontend_key:
                self.frontends[frontend_key] = self.make_mlp(mlp_in_features=cnn_features)
                self.losses_and_scores[f"{frontend_key}_bce_loss"] = torch.nn.functional.binary_cross_entropy_with_logits
                self.losses_and_scores[f"{frontend_key}_f1_score"] = sklearn.metrics.f1_score
            else:
                raise KeyError
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        print(f"  CNN encoder parameters {cnn_parameters:,}")

        for key, frontend_key in self.frontends.items():
            n_parameters = sum(p.numel() for p in frontend_key.parameters() if p.requires_grad)
            print(f"  Frontend `{key}` parameters: {n_parameters:,}")

        self.example_input_array = torch.zeros(
            (4, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )

        self.initialize_layers()

    def setup(self, stage=None):
        datamodule = self.trainer.datamodule
        for label_percentile in ['label_scaled_25p', 'label_scaled_50p', 'label_scaled_75p']:
            assert hasattr(datamodule, label_percentile)
            setattr(self, label_percentile, getattr(datamodule, label_percentile))

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            min_lr=2e-5,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            verbose=True,
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': self.monitor_metric,
        }

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        results = {}
        features = self.cnn_encoder(signals)
        for frontend_key, frontend in self.frontends.items():
            results[frontend_key] = frontend(features)
        return results

    def update_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        signals, labels, class_labels_50p, class_labels_25p, class_labels_75p = batch
        # print(f"  min {labels.min()}, max {labels.max()} (stage {stage} batch {batch_idx})")
        results = self(signals)
        sum_loss = None
        for frontend_key in self.frontends.keys():
            frontend_result = results[frontend_key]
            if 'time_to_elm' in frontend_key:
                target = labels
            elif 'reconstruction' in frontend_key:
                target = signals
            elif frontend_key == 'classifier_25_mlp':
                target = class_labels_25p
            elif frontend_key == 'classifier_50_mlp':
                target = class_labels_50p
            elif frontend_key == 'classifier_75_mlp':
                target = class_labels_75p
            else:
                raise KeyError
            for loss_or_score_name, func in self.losses_and_scores.items():
                if frontend_key not in loss_or_score_name:
                    continue
                if 'loss' in loss_or_score_name:
                    metric_value = func(
                        input=frontend_result,
                        target=target.type_as(frontend_result),
                    )
                    sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
                elif 'score' in loss_or_score_name:
                    kwargs = {}
                    if 'f1' in loss_or_score_name:
                        modified_predictions = (frontend_result > 0.5).type(torch.int)
                        kwargs['zero_division'] = 0
                    else:
                        modified_predictions = frontend_result
                    metric_value = func(
                        y_pred=modified_predictions.detach().cpu(), 
                        y_true=target.detach().cpu(),
                        **kwargs,
                    )
                self.log(f"{loss_or_score_name}/{stage}", metric_value, sync_dist=True)
        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)
        return sum_loss

    def on_fit_start(self) -> None:
        self.t_fit_start = time.time()

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        print(f"Epoch {self.current_epoch} start")
        for frontend_key, frontend in self.frontends.items():
            if self.current_epoch < self.unfreeze_epoch[frontend_key]:
                print(f'  Frontend `{frontend_key}` is frozen')
                frontend.freeze()
            else:
                print(f'  Frontend `{frontend_key}` is not frozen')
                frontend.unfreeze()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.update_step(batch, batch_idx, stage='train')

    def on_train_epoch_end(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' not in name:
                continue
            values = param.data.detach()
            mean = torch.mean(values).item()
            std = torch.std(values).item()
            z_scores = (values-mean)/std
            skew = torch.mean(z_scores**3).item()
            kurt = torch.mean(z_scores**4).item()
            self.log(f"param_mean/{name}", mean, sync_dist=True)
            self.log(f"param_std/{name}", std, sync_dist=True)
            self.log(f"param_skew/{name}", skew, sync_dist=True)
            self.log(f"param_kurt/{name}", kurt, sync_dist=True)
        print(f"Epoch {self.current_epoch} elapsed train time: {(time.time()-self.t_train_epoch_start)/60:0.1f} min")

    def on_validation_epoch_start(self) -> None:
        self.t_val_epoch_start = time.time()

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='val')

    def on_validation_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch} elapsed valid. time: {(time.time()-self.t_val_epoch_start)/60:0.1f} min")

    def on_fit_end(self) -> None:
        print(f"Fit elapsed time {(time.time()-self.t_fit_start)/60:0.1f} min")

    def on_test_start(self) -> None:
        self.t_test_start = time.time()

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='test')

    def on_test_epoch_end(self) -> None:
        print(f"Test elapsed time {(time.time()-self.t_test_start)/60:0.1f} min")

    def on_predict_start(self) -> None:
        self.predict_outputs: list[list] = []
        self.t_predict_start = time.time()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signals, labels, class_labels, shot, elm_index, t0 = batch
        results = self(signals)
        if batch_idx == 0:
            self.predict_outputs.append([])
            assert dataloader_idx == len(self.predict_outputs)-1
        prediction_outputs = {
            'labels': labels,
            'signals': signals,
            'class_labels': class_labels,
            'shot': shot,
            'elm_index': elm_index,
            't0': t0,
        }
        prediction_outputs.update(results)
        self.predict_outputs[-1].append(prediction_outputs)
        return True
    
    def on_predict_epoch_end(self) -> None:
        if self.global_rank != 0:
            return
        if 'time_to_elm_mlp' in self.predict_outputs[0][0]:
            i_page = 1
            for i_elm, result in enumerate(self.predict_outputs):
                shot = result[0]['shot'][0]
                elm_index = result[0]['elm_index'][0]
                t0 = result[0]['t0'][0]
                labels: torch.Tensor = torch.concat([batch['labels'] for batch in result]).squeeze().numpy(force=True)
                predictions: torch.Tensor = torch.concat([batch['time_to_elm_mlp'] for batch in result]).squeeze().numpy(force=True)
                signals: torch.Tensor = torch.concat([batch['signals'] for batch in result]).squeeze().numpy(force=True)
                assert labels.shape[0] == predictions.shape[0] and labels.shape[0] == signals.shape[0]
                if i_elm % 6 == 0:
                    plt.close('all')
                    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
                    plt.suptitle(f"Inference on ELMs in test dataset (page {i_page})")
                signal = signals[:, -1, 2, 3].squeeze()
                pre_elm_size = np.count_nonzero(np.isfinite(labels))
                time = (np.arange(len(labels)) - pre_elm_size) / 1e3
                plt.sca(axes.flat[i_elm%6])
                plt.plot(time, labels, label='Label')
                plt.plot(time, predictions, label='Prediction')
                plt.ylabel("Label | Prediction")
                plt.xlabel('Time to ELM (ms)')
                plt.title(f'ELM index {elm_index} | Shot {shot} @ {t0:.1f} ms')
                plt.legend(fontsize='small', loc='upper right')
                twinx = axes.flat[i_elm%6].twinx()
                twinx.plot(time, signal, label='Signal', color='C2')
                twinx.set_ylabel('Scaled signal')
                twinx.legend(fontsize='small', loc='lower right')
                # print(f"  min {np.nanmin(labels)}, max {np.nanmax(labels)}")
                if i_elm % 6 == 5 or i_elm == len(self.predict_outputs)-1:
                    plt.tight_layout()
                    filename = f'inference_{i_page:02d}'
                    filepath = os.path.join(self.log_dir, filename)
                    print(f"Saving figures {filepath}{{.pdf,.png}}")
                    plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                    plt.savefig(filepath+'.png', format='png', transparent=True)
                    for logger in self.loggers:
                        if isinstance(logger, loggers.TensorBoardLogger):
                            logger.experiment.add_figure(f"inference/{filename}", fig, close=False)
                        elif isinstance(logger, loggers.WandbLogger):
                            logger.log_image(key='inference', images=[filepath+'.png'])
                    i_page += 1
                    plt.close('all')

        if 'reconstruction_decoder' in self.predict_outputs[0][0]:
            plt.set_cmap('seismic')
            for i_elm, result in enumerate(self.predict_outputs):
                shot = result[0]['shot'][0]
                elm_index = result[0]['elm_index'][0]
                t0 = result[0]['t0'][0]
                class_labels: torch.Tensor = torch.concat([batch['class_labels'] for batch in result]).squeeze().numpy(force=True)
                reconstruction: torch.Tensor = torch.concat([batch['reconstruction_decoder'] for batch in result]).squeeze().numpy(force=True)
                signals: torch.Tensor = torch.concat([batch['signals'] for batch in result]).squeeze().numpy(force=True)
                assert class_labels.shape[0] == reconstruction.shape[0] and class_labels.shape[0] == signals.shape[0]
                assert np.array_equiv(tuple(reconstruction.shape), tuple(signals.shape))
                pre_elm_size = np.flatnonzero(class_labels == 1)[0]  # length of pre-ELM phase
                pre_elm_t0 = pre_elm_size - self.signal_window_size
                t0_array = [0, pre_elm_t0//3, 2*pre_elm_t0//3, pre_elm_t0]
                fig, axes = plt.subplots(
                    ncols=4, 
                    nrows=4, 
                    figsize=(12, 8),
                    sharex='col',
                    sharey='row',
                )
                plt.suptitle(f"Autoencoder reconstructions | ELM index {elm_index} | shot {shot} @ {t0:.1f} ms")
                i_radial_row = 3
                i_poloidal_column = 5
                for i, t0 in enumerate(t0_array):
                    plt.sca(axes.flat[i])
                    plt.imshow(
                        signals[t0, :, i_radial_row, :].T,
                        origin='lower',
                        aspect='auto', vmin=-4, vmax=4,
                    )
                    plt.title(f'Signal | t={t0}')
                    if i==0:
                        plt.ylabel(f'Radial row {i_radial_row+1}')
                    plt.sca(axes.flat[i+4])
                    plt.imshow(
                        reconstruction[t0, :, i_radial_row, :].T,
                        origin='lower',
                        aspect='auto', vmin=-4, vmax=4,
                    )
                    plt.title(f'Reconstruction | t={t0}')
                    if i==0:
                        plt.ylabel(f'Radial row {i_radial_row+1}')
                    plt.sca(axes.flat[i+8])
                    plt.imshow(
                        signals[t0, :, :, i_poloidal_column].T,
                        aspect='auto', vmin=-4, vmax=4,
                    )
                    plt.title(f'Signal | t={t0}')
                    if i==0:
                        plt.ylabel(f'Poloidal column {i_poloidal_column+1}')
                    plt.sca(axes.flat[i+12])
                    plt.imshow(
                        reconstruction[t0, :, :, i_poloidal_column].T,
                        aspect='auto', vmin=-4, vmax=4,
                    )
                    plt.title(f'Reconstruction | t={t0}')
                    if i==0:
                        plt.ylabel(f'Poloidal column {i_poloidal_column+1}')
                    plt.xlabel('Time (mu-s)')
                plt.tight_layout()
                filename = f'reconstruction_{elm_index:04d}'
                filepath = os.path.join(self.log_dir, filename)
                print(f"Saving figures {filepath}{{.pdf,.png}}")
                plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'.png', format='png', transparent=True)
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(f"inference/{filename}", fig, close=False)
                    elif isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='inference', images=[filepath+'.png'])
                plt.close()

    def on_predict_end(self) -> None:
        print(f"Predict elapsed time {(time.time()-self.t_predict_start)/60:0.1f} min")
