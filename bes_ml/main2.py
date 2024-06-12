from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
from logging import Logger
from typing import \
    OrderedDict, Iterable, Mapping, Callable, Any, Sequence, Sized
import os
import time


import numpy as np
import scipy.stats
import scipy.signal
import sklearn.metrics
import h5py
import wandb

import torch
import torch.nn
import torch.utils.data
import torch.cuda
import torch.optim
import torch.optim.lr_scheduler

from lightning.pytorch import Trainer, LightningModule, LightningDataModule, Callback
from lightning.pytorch.strategies import Strategy, DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import \
    LearningRateMonitor, EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)


def print_fields(obj):
    print(f"{obj.__class__.__name__} fields:")
    class_fields_dict = {field.name: field for field in dataclasses.fields(obj.__class__)}
    for field_name in dataclasses.asdict(obj):
        value = getattr(obj, field_name)
        field_str = f"  {field_name}: {value}"
        default_value = class_fields_dict[field_name].default
        if value != default_value:
            field_str += f" (default {default_value})"
        print(field_str)


@dataclasses.dataclass(eq=False)
class _Base_Class:
    signal_window_size: int = 1024
    is_global_zero: bool = False

    def __post_init__(self):
        assert np.log2(self.signal_window_size).is_integer(), \
            'Signal window must be power of 2'


@dataclasses.dataclass(eq=False)
class Model(LightningModule, _Base_Class):
    initial_max_lr: float = 1e-3  # maximum LR used by first layer
    layerwise_lr_decrement: float = 1.5
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    leaky_relu_slope: float = 1e-2
    monitor_metric: str = 'sum_loss/val'

    def __post_init__(self):

        # init superclasses
        super().__init__()
        super(LightningModule, self).__post_init__()

        self.save_hyperparameters()
        if self.is_global_zero:
            print_fields(self)

        # input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.feature_model, self.feature_space_size = self.make_feature_model()

        # task sub-models and metrics
        self.task_models = torch.nn.ModuleDict()
        self.task_models_layers = {}
        self.task_metrics: dict[str, dict] = {}

        # binary classifier task
        task_name = 'median_classifier'
        self.task_models[task_name] = self.make_mlp_classifier()
        self.task_metrics[task_name] = {
            'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
            'f1_score': sklearn.metrics.f1_score,
        }

        self.total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.is_global_zero: print(f"Total model parameters: {self.total_parameters:,}")

        if self.is_global_zero: print("Initializing model to uniform random weights and biases=0")
        for name, param in self.named_parameters():
            if 'bn' in name: continue
            if name.endswith("bias"):
                if self.is_global_zero: print(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith("weight"):
                n_in = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / n_in)
                param.data.uniform_(-sqrt_k, sqrt_k)
                if self.is_global_zero: print(f"  {name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")
            else:
                raise ValueError

        if self.is_global_zero: 
            print("Batch evaluation (batch_size=128) with randn() data")
            self.example_batch_data = torch.randn(
                size=[128]+list(self.input_data_shape[1:]),
                dtype=torch.float32,
            )
            example_batch_output = self(self.example_batch_data)
            for task_name, task_output in example_batch_output.items():
                print(f"  {task_name} output shape: {task_output.shape}  mean: {torch.mean(task_output):.3e}  var: {torch.var(task_output):.3e}")

    def make_feature_model(self) -> tuple[torch.nn.Module, int]:

        if self.is_global_zero: print("Feature space sub-model")

        feature_layer_dict = OrderedDict()

        conv_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 8, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 8, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 8, 'kernel': (1, 4, 4), 'stride': 1},
        )

        data_shape = self.input_data_shape
        if self.is_global_zero: print(f"  Data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels: int|Any = None
        for i_layer, layer in enumerate(conv_layers):
            if i_layer != 0:
                feature_layer_dict[f"bn_{i_layer:02d}"] = torch.nn.BatchNorm3d(
                    num_features=out_channels,
                )
            layer_name = f"conv_{i_layer:02d}"
            conv = torch.nn.Conv3d(
                in_channels=1 if out_channels is None else out_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                bias=False,
            )
            n_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            if self.is_global_zero: 
                print(f"  {layer_name} kern {conv.kernel_size}  stride {conv.stride}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {np.prod(data_shape)})")
            out_channels = conv.out_channels
            feature_layer_dict[layer_name] = conv

        feature_model = torch.nn.Sequential(feature_layer_dict)

        output_size = feature_model(torch.zeros(self.input_data_shape)).numel()
        if self.is_global_zero: print(f"  Feature space size: {output_size}")
        assert output_size == 128

        n_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
        if self.is_global_zero: print(f"  Feature sub-model parameters: {n_params:,d}")

        return feature_model, output_size

    def make_mlp_classifier(self) -> torch.nn.Module:

        if self.is_global_zero: print("MLP classifier sub-model")

        mlp_layer_dict = OrderedDict()

        assert self.feature_space_size
        mlp_layers = (self.feature_space_size, 64, 32, 1)

        for i in range(len(mlp_layers)-1):
            mlp_layer_dict[f"bn_{i:02d}"] = torch.nn.BatchNorm1d(
                num_features=mlp_layers[i],
            )
            layer_name = f"fc_{i:02d}"
            fc_layer = torch.nn.Linear(
                in_features=mlp_layers[i],
                out_features=mlp_layers[i+1],
                # bias=True if (i+1) < len(mlp_layers)-1 else False,
                bias=False,
            )
            n_params = sum(p.numel() for p in fc_layer.parameters() if p.requires_grad)
            if self.is_global_zero: 
                print(f"  {layer_name}  in_features {fc_layer.in_features}  out_features {fc_layer.out_features}  parameters {n_params:,d}")
            mlp_layer_dict[layer_name] = fc_layer

        mlp_classifier = torch.nn.Sequential(mlp_layer_dict)

        n_params = n_params = sum(p.numel() for p in mlp_classifier.parameters() if p.requires_grad)
        if self.is_global_zero: print(f"  MLP sub-model parameters: {n_params:,d}")

        return mlp_classifier

    def configure_optimizers(self):
        parameter_group = []
        lr = self.initial_max_lr
        if self.is_global_zero: print("Initial layer-wise learning rates")
        for layer_name, layer in self.feature_model.named_children():
            if 'bn' in layer_name:
                for param_name, param in layer.named_parameters():
                    parameter_group.append({
                        'params': param,
                        'lr': self.initial_max_lr/10,
                    })
            else:
                for param_name, param in layer.named_parameters():
                    assert param_name.endswith('weight') or param_name.endswith('bias')
                    param_lr = lr if param_name.endswith('weight') else lr/8
                    parameter_group.append({
                        'params': param,
                        'lr': param_lr,
                    })
                    if self.is_global_zero: print(f"  {layer_name} {param_name} {param_lr:.3e}")
                lr /= self.layerwise_lr_decrement
        lr_after_feature_model = lr
        for task_name, task_model in self.task_models.items():
            lr = lr_after_feature_model
            for layer_name, layer in task_model.named_children():
                if 'bn' in layer_name:
                    for param_name, param in layer.named_parameters():
                        parameter_group.append({
                            'params': param,
                            'lr': self.initial_max_lr/10,
                        })
                else:
                    for param_name, param in layer.named_parameters():
                        assert param_name.endswith('weight') or param_name.endswith('bias')
                        param_lr = lr if param_name.endswith('weight') else lr/8
                        parameter_group.append({
                            'params': param,
                            'lr': param_lr,
                        })
                        if self.is_global_zero: print(f"  {task_name} {layer_name} {param_name} {param_lr:.3e}")
                    lr /= self.layerwise_lr_decrement

        self.optimizer = torch.optim.Adam(
            parameter_group,
            lr=self.initial_max_lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            mode='min' if 'loss' in self.monitor_metric else 'max',
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'monitor': self.monitor_metric,
            },
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for layer in self.feature_model.children():
            x = torch.nn.functional.leaky_relu(layer(x), negative_slope=self.leaky_relu_slope)
        features = x.flatten(1)
        results = {}
        for task_model_name, task_model in self.task_models.items():
            x = features
            children_layers = list(task_model.children())
            nlayers = len(children_layers)
            for i, layer in enumerate(children_layers):
                x = layer(x)
                if i+1 < nlayers:
                    x = torch.nn.functional.leaky_relu(x, negative_slope=self.leaky_relu_slope)
            results[task_model_name] = x
        return results

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.update_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='test')

    def update_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        signal_window, time_to_elm, quantiles = batch
        task_results = self(signal_window)
        sum_loss = torch.Tensor([0.0])
        for task, task_metrics in self.task_metrics.items():
            results: torch.Tensor = task_results[task]
            if 'class' in task:
                labels: torch.Tensor = quantiles[0.5]
            for metric_name, metric_function in task_metrics.items():
                if 'loss' in metric_name:
                    metric_value = metric_function(
                        input=results.reshape_as(labels),
                        target=labels.type_as(results),
                    )
                    sum_loss = sum_loss + metric_value if sum_loss else metric_value
                elif 'score' in metric_name:
                    kwargs = {}
                    if 'f1' in metric_name:
                        modified_predictions = (results > 0.5).type(torch.int)
                        kwargs['zero_division'] = 0
                    else:
                        modified_predictions = results
                    metric_value = metric_function(
                        y_pred=modified_predictions.detach().cpu(), 
                        y_true=labels.detach().cpu(),
                        **kwargs,
                    )
                self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True)
        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)
        return sum_loss

    def on_fit_start(self):
        self.t_fit_start = time.time()
        if self.is_global_zero:
            print(f"Fit start with global step")

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        if self.is_global_zero:
            print(f"Fit time: {delt/60:0.1f} min")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.t_train_epoch_start
        global_time = time.time() - self.t_fit_start
        epoch_steps = self.global_step-self.s_train_epoch_start
        if self.is_global_zero and self.global_step > 0:
            logged_metrics = self.trainer.logged_metrics
            line =  f"  Ep {self.current_epoch:03d}  "
            line += f"train/val loss {logged_metrics['sum_loss/train']:.3f}/{logged_metrics['sum_loss/val']:.3f}  "
            line += f"ep/gl steps {epoch_steps:,d}/{self.global_step:,d}  "
            line += f"ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}  " 
            print(line)

    def setup(self, stage=None):  # fit, validate, test, or predict
        assert self.is_global_zero == self.trainer.is_global_zero
        if self.is_global_zero:
            assert self.global_rank == 0


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signal_windows: dict|Any = None # global signal window data mapping to dataset index
    signal_window_size: int = 0
    elm_split_for_rank: Iterable = () # rank-wise ELM indices
    shot_split_for_rank: Iterable = () # rank-wise shots
    signals_for_rank: dict|Any = None # rank-wise signals (map to ELM indices)
    quantile_min: float|Any = None
    quantile_max: float|Any = None
    contrastive_learning: bool = False
    time_to_elm_quantiles: dict[float, float]|Any = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        # self.t0_indices = list(self.t0_and_time_to_elm_labels.keys())
        # self.time_to_elm_labels = list(self.t0_and_time_to_elm_labels.values())
        # if self.time_to_elm_quantiles:
        #     if self.is_global_zero:
        #         print("  Using input time-to-ELM quantiles")
        # else:
        #     if self.is_global_zero:
        #         print("  Calculating time-to-ELM quantiles")
        #     quantiles = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        #     quantile_values = np.quantile(self.time_to_elm_labels, quantiles)
        #     self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
        # if self.is_global_zero: 
        #     print(f"  Time-to-ELM quantiles for binary labels:")
        #     for q, qval in self.time_to_elm_quantiles.items():
        #         print(f"    Quantile {q:.2f}: {qval:.1f} ms")

        # restrict quantile range
        if self.quantile_min is not None and self.quantile_max is not None:
            qmin_val, qmax_val = np.quantile(self.time_to_elm_labels, (self.quantile_min, self.quantile_max))
            if not self.contrastive_learning:
                if self.is_global_zero: 
                    print(f"  Restricting time-to-ELM labels to quantile range: {self.quantile_min:.2f}-{self.quantile_max:.2f}")
                mask = np.logical_and(
                    self.time_to_elm_labels >= qmin_val,
                    self.time_to_elm_labels <= qmax_val,
                )
            else:
                if self.is_global_zero: 
                    print(f"  Contrastive learning with time-to-ELM quantiles 0.0-{self.quantile_min:.2f} and {self.quantile_max:.2f}-1.0")
                mask = np.logical_or(
                    self.time_to_elm_labels <= qmin_val,
                    self.time_to_elm_labels >= qmax_val,
                )
            self.time_to_elm_labels = np.array(self.time_to_elm_labels)[mask].tolist()
            self.t0_indices = np.array(self.t0_indices, dtype=int)[mask].tolist()
            if self.is_global_zero:
                print(f"  Restricted time-to-ELM min/max: {np.min(self.time_to_elm_labels):.1f}-{np.max(self.time_to_elm_labels):.1f} ms")
                print(f"  Restricted data signal windows: {len(self):,d}")

    def __len__(self) -> int:
        return len(self.t0_indices)
    
    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.t0_indices[i]
        time_to_elm = self.time_to_elm_labels[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        quantile_binary_label = {q: int(time_to_elm<=qval) for q, qval in self.time_to_elm_quantiles.items()}
        return signal_window, time_to_elm, quantile_binary_label


@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    data_file: str|Path|Any = None
    max_elms: int|Any = None
    batch_size: int = 128
    stride_factor: int = 8
    num_workers: int = 0
    outlier_value: float = 6
    fraction_validation: float = 0.2
    fraction_test: float = 0.2
    use_random_data: bool = False
    seed: int = 0  # seed for ELM index shuffling; must be same across processes
    quantile_min: float|Any = None
    quantile_max: float|Any = None
    contrastive_learing: bool = False

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.data_file = Path(self.data_file).absolute()
        assert self.data_file.exists()

        self.datasets: dict = {}
        self.shot_split: dict[str,Sequence] = {}
        self.shot_split_by_rank: dict[str,Sequence] = {}
        self.elm_split: dict[str,Sequence] = {}
        self.elm_split_by_rank: dict[str,Sequence] = {}
        self.time_to_elm_quantiles: dict[float,float] = {}

        self.trainer: Trainer|Any = None
        self.raw_signal_mean: float|Any = None
        self.raw_signal_stdev: float|Any = None

        if self.is_global_zero:
            print_fields(self)

        # datamodule state, to reproduce pre-processing
        self.state_items = [
            'raw_signal_mean',
            'raw_signal_stdev',
            'shot_split',
            'elm_split',
            'time_to_elm_quantiles',
        ]
        for item in self.state_items:
            assert hasattr(self, item)

    def prepare_data(self):
        assert self.batch_size % self.trainer.world_size == 0
        self.batch_size_per_rank = self.batch_size//self.trainer.world_size

    def setup(self, stage: str):
        assert stage in ['fit', 'test', 'predict']
        assert self.is_global_zero == self.trainer.is_global_zero

        print(f"Data setup on rank {self.trainer.global_rank} (world size {self.trainer.world_size})")
        if self.is_global_zero: 
            print(f"Batch size: {self.batch_size} ({self.batch_size_per_rank} per rank)")

        if 'train' not in self.elm_split or not self.elm_split['train']:
            self._get_elm_indices_and_split()

        stages = ['train', 'validation'] if stage == 'fit' else [stage]
        for st in stages:
            assert st in ['train', 'validation', 'test', 'predict']
            if st in self.datasets and isinstance(self.datasets[st], torch.utils.data.Dataset):
                continue
            indices = self.elm_split[st]
            assert indices is not None
            if self.is_global_zero: 
                print(f"Reading {len(indices)} ELMs for stage {st}")
            signal_windows = []
            outliers = 0
            with h5py.File(self.data_file, 'r') as h5_file:
                elms: h5py.Group = h5_file['elms']
                for i_elm, elm_index in enumerate(indices):
                    if i_elm%100 == 0 and self.is_global_zero:
                        print(f"  Reading ELM event {i_elm:04d}/{len(indices):04d}")
                    elm_event: h5py.Group = elms[f"{elm_index:06d}"]
                    shot = int(elm_event.attrs['shot'])
                    assert elm_event["bes_signals"].shape[0] == 64
                    assert elm_event['bes_time'].size == elm_event["bes_signals"].shape[1]
                    time = np.array(elm_event['bes_time'], dtype=np.float32)
                    t_start: float = elm_event.attrs['t_start']
                    i_start: int = np.flatnonzero(time >= t_start)[0]
                    t_stop: float = elm_event.attrs['t_stop'] - 0.05
                    i_stop: int = np.flatnonzero(time <= t_stop)[-1]
                    i_window_stop = i_stop
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                    while True:
                        i_window_start = i_window_stop - self.signal_window_size
                        if i_window_start < i_start: break
                        if self.outlier_value:
                            signal_window = signals[i_window_start:i_window_stop, ...]
                            assert signal_window.shape[0] == self.signal_window_size
                            if np.abs(signal_window).max() > self.outlier_value:
                                i_window_stop -= self.signal_window_size // self.stride_factor
                                outliers += 1
                                continue
                        signal_windows.append({
                            'elm_index': elm_index,
                            'shot': shot,
                            'i_t0': i_window_start,
                            'time_to_elm': time[i_stop] - time[i_window_stop]
                        })
                        i_window_stop -= self.signal_window_size // self.stride_factor
            
            n_signal_windows = len(signal_windows)
            if self.is_global_zero: 
                print(f"  Signal windows: {n_signal_windows:,d}  ({outliers:,d} outliers removed)")
                print(f"  Steps per epoch {n_signal_windows/self.batch_size:,.1f}")

            # Raw signal stats
            self._get_statistics(signal_windows=signal_windows, stage=st)
            assert self.raw_signal_mean and self.raw_signal_stdev and self.time_to_elm_quantiles
            if st == 'train':
                self.save_hyperparameters({
                    'raw_signal_mean': self.raw_signal_mean,
                    'raw_signal_stdev': self.raw_signal_stdev,
                    'time_to_elm_quantiles': self.time_to_elm_quantiles,
                })

            # get rank-wise shot signals
            signals_for_rank = {}
            with h5py.File(self.data_file) as root:
                elm_list_for_rank = self.elm_split_by_rank[st][self.trainer.global_rank]
                for elm_index in elm_list_for_rank:
                    elm_group = root['elms'][f"{elm_index:06d}"]
                    signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                    # normalized signals
                    signals_for_rank[elm_index] = (signals - self.raw_signal_mean) / self.raw_signal_stdev

            # rank-wise datasets
            if st in ['train', 'validation', 'test']:
                self.datasets[st] = ELM_TrainValTest_Dataset(
                    signal_windows=signal_windows,
                    signal_window_size=self.signal_window_size,
                    elm_split_for_rank=self.elm_split_by_rank[st][self.trainer.global_rank],
                    shot_split_for_rank=self.shot_split_by_rank[st][self.trainer.global_rank],
                    signals_for_rank=signals_for_rank,
                    # rank=self.trainer.global_rank,
                    quantile_min=self.quantile_min,
                    quantile_max=self.quantile_max,
                    contrastive_learning=self.contrastive_learing,
                    time_to_elm_quantiles=self.time_to_elm_quantiles,
                )
            
            if st in ['test', 'predict']:
                pass

    def _get_elm_indices_and_split(self):
        rng = np.random.default_rng(self.seed)
        with h5py.File(self.data_file, 'r') as root:
            shots = sorted([int(shot_key) for shot_key in root['shots']])
            assert len(shots) == len(set(shots))
            shots = set(shots)
            # elms_with_shots = {int(elm_key): elm_group.attrs['shot'] for elm_key, elm_group in root['elms'].items()}
            shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(shots ^ shots_from_elms) == 0
            elms = [int(elm_key) for elm_key in root['elms']]
            # shuffle ELM indices
            if self.is_global_zero: 
                print(f"Total ELMs in dataset: {len(elms)}")
                print(f"Total shots in dataset: {len(shots)}")
            # limit max ELMs
            if self.max_elms and len(elms) > self.max_elms:
                elms = elms[:self.max_elms]
                shots = set([int(root['elms'][f"{elm_index:06d}"].attrs['shot']) for elm_index in elms])
                if self.is_global_zero:
                    print(f"ELMs in use: {len(elms)}")
                    print(f"Shots in use: {len(shots)}")
            # shuffle shots in dataset
            print(f"Shuffling sorted shots with seed={self.seed} (rank {self.trainer.global_rank})")
            shots = np.array(list(shots), dtype=int)
            rng.shuffle(shots)
            # split shots
            n_test_shots = int(self.fraction_test * shots.size)
            n_validation_shots = int(self.fraction_validation * shots.size)
            self.shot_split['test'], self.shot_split['validation'], self.shot_split['train'] = \
                np.split(shots, [n_test_shots, n_test_shots+n_validation_shots])
            self.shot_split_by_rank = {}
            for stage in ['train', 'validation', 'test']:
                print(f" Stage {stage} with {self.shot_split[stage].size} shots ({self.shot_split[stage].size/shots.size*1e2:.1f}%)")
                self.shot_split_by_rank[stage] = np.array_split(self.shot_split[stage], self.trainer.world_size)
                assert len(self.shot_split_by_rank[stage]) == self.trainer.world_size
                self.elm_split_by_rank[stage] = [
                    [int(key) for key, value in root['elms'].items() if value.attrs['shot'] in rank_shot_list]
                    for rank_shot_list in self.shot_split_by_rank[stage]
                ]
                assert len(self.elm_split_by_rank[stage]) == self.trainer.world_size
                self.elm_split[stage] = list(set().union(*self.elm_split_by_rank[stage]))
                for i_rank in range(len(self.elm_split_by_rank[stage])):
                    rng.shuffle(self.elm_split_by_rank[stage][i_rank])

        if self.is_global_zero: 
            print("ELMs for analysis")
        for stage, elm_indices in self.elm_split.items():
            if self.is_global_zero: 
                print(f"  {stage} ELMs: {len(elm_indices)}  ({len(elm_indices)/len(elms)*1e2:.1f}%)")

    def _get_statistics(
            self, 
            signal_windows: list[dict],
            stage: str,
    ) -> dict:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_interval = np.max([self.stride_factor, len(signal_windows)//int(50e3)])
        last_elm_index = -1
        with h5py.File(self.data_file) as root:
            for elm_dict in signal_windows[::stat_interval]:
                elm_index = elm_dict['elm_index']
                if elm_index != last_elm_index:
                    elm_event = root['elms'][f'{elm_index:06d}']
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                last_elm_index = elm_index
                i_t0 = elm_dict['i_t0']
                signal_window = signals[i_t0: i_t0 + self.signal_window_size, :, :]
                assert signal_window.shape[0] == self.signal_window_size
                signal_min = np.min([signal_min, signal_window.min()])
                signal_max = np.max([signal_max, signal_window.max()])
                hist, bin_edges = np.histogram(
                    signal_window,
                    bins=n_bins,
                    range=(-10.4, 10.4),
                )
                cummulative_hist += hist
        bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
        stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
        exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
        if self.is_global_zero: 
            print(f"  Raw signals min {signal_min:.2f} max {signal_max:.2f} mean {mean:.2f} stdev {stdev:.2f} exkurt {exkurt:.2f}")
        # time-to-ELM quantiles
        time_to_elm_list = [e['time_to_elm'] for e in signal_windows]
        quantiles = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        quantile_values = np.quantile(time_to_elm_list, quantiles)
        if stage == 'train':
            self.raw_signal_mean = mean.item()
            self.raw_signal_stdev = stdev.item()
            self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}

    def get_state_dict(self) -> dict:
        state_dict = {
            item: getattr(self, item) for item in self.state_items
        }
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])
            if self.is_global_zero:
                print(f"Loading state item {item} = {getattr(self, item)}")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('train')

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('validation')

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('test')

    def predict_dataloader(self) -> None:
        pass
        # return [
        #     torch.utils.data.DataLoader(
        #         dataset=dataset,
        #         batch_size=self.batch_size_per_rank,
        #         num_workers=self.num_workers,
        #         persistent_workers=True,
        #     ) for dataset in self.datasets['predict']
        # ]

    def _train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        is_distributed = self.trainer.world_size > 1
        shuffle = True if stage=='train' else False
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.datasets[stage],
            shuffle=shuffle,
        ) if is_distributed else None
        return torch.utils.data.DataLoader(
            dataset=self.datasets[stage],
            sampler=sampler,
            batch_size=self.batch_size_per_rank,  # batch size per rank
            num_workers=self.num_workers,
            shuffle=None if is_distributed else shuffle,
            # prefetch_factor=2 if self.num_workers else None,
            # persistent_workers=bool(self.num_workers),
            pin_memory=True,
        )


def main(
        data_file: str|Path,
        signal_window_size = 1024,
        experiment_name = 'experiment_default',
        # model
        initial_max_lr = 1e-3,
        layerwise_lr_decrement = 1.5,
        weight_decay = 1e-4,
        # loggers
        log_freq = 50,
        use_wandb = False,
        # callbacks
        early_stopping_min_delta = 1e-3,
        early_stopping_patience = 5,
        # trainer
        max_epochs = 2,
        gradient_clip_val = 500,
        batch_size = 64,
        # data
        max_elms = 100,
        fraction_validation = 0.1,
        fraction_test = 0,
        num_workers = 0,
        quantile_min = 0.4,
        quantile_max = 0.6,
        contrastive_learning = True,
):

    # SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    rank = int(os.getenv("SLURM_PROCID", default=0))
    local_rank = int(os.getenv("SLURM_LOCALID", default=0))
    node_rank = int(os.getenv("SLURM_NODEID", default=0))

    is_global_zero = rank == 0
    if is_global_zero:
        print(f"World size {world_size} on {num_nodes} node(s)")
    print(f"Rank {rank} of world size {world_size} (local rank {local_rank} on node {node_rank})")

    ### model
    lit_model = Model(
        signal_window_size=signal_window_size,
        initial_max_lr=initial_max_lr,
        layerwise_lr_decrement=layerwise_lr_decrement,
        weight_decay=weight_decay,
        is_global_zero=is_global_zero,
    )
    if is_global_zero:
        print("Model Summary:")
        print(ModelSummary(lit_model, max_depth=-1))

    ### callbacks
    monitor_metric = lit_model.monitor_metric
    metric_mode = 'min' if 'loss' in monitor_metric else 'max'
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor=monitor_metric,
            mode=metric_mode,
            save_last=True,
        ),
        # DeviceStatsMonitor(),
        EarlyStopping(
            monitor=monitor_metric,
            mode=metric_mode,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            log_rank_zero_only=True,
            verbose=True,
        ),
    ]

    ### loggers
    loggers = []
    experiment_dir = Path(experiment_name).absolute()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
    trial_name = f"r{slurm_identifier}_{datetime_str}" if slurm_identifier else f"r{datetime_str}"
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir.parent,
        name=experiment_name,
        version=trial_name,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    trial_dir = Path(tb_logger.log_dir).absolute()
    if is_global_zero:
        print(f"Trial directory: {trial_dir}")
    if use_wandb:
        wandb.login()
        wandb_logger = WandbLogger(
            save_dir=experiment_dir,
            project=experiment_name,
            name=trial_name,
        )
        wandb_logger.watch(
            model=lit_model, 
            log='all', 
            log_freq=log_freq,
        )
        loggers.append(wandb_logger)

    ### initialize trainer
    trainer = Trainer(
        max_epochs = max_epochs,
        gradient_clip_val = gradient_clip_val,
        logger = loggers,
        log_every_n_steps = log_freq,
        callbacks = callbacks,
        enable_checkpointing = True,
        enable_progress_bar = False,
        enable_model_summary = False,
        precision = '16-mixed' if torch.cuda.is_available() else 32,
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=True,
        ) if world_size>1 else 'auto',
        use_distributed_sampler = world_size>1,
        num_nodes = num_nodes,
    )

    assert trainer.node_rank == node_rank
    assert trainer.world_size == world_size
    assert trainer.local_rank == local_rank
    assert trainer.global_rank == rank
    assert trainer.is_global_zero == is_global_zero

    ### data
    lit_datamodule = Data(
        signal_window_size=signal_window_size,
        data_file=data_file,
        max_elms=max_elms,
        batch_size=batch_size,
        fraction_test=fraction_test,
        fraction_validation=fraction_validation,
        num_workers=num_workers,
        quantile_min=quantile_min,
        quantile_max=quantile_max,
        contrastive_learing=contrastive_learning,
        is_global_zero=is_global_zero,
    )

    trainer.fit(lit_model, datamodule=lit_datamodule)

    if fraction_test:
        trainer.test(lit_model, lit_datamodule)

    if use_wandb:
        wandb.finish()

if __name__=='__main__':
    main(
        # data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        data_file='/Users/drsmith/Documents/repos/bes-ml/bes_ml/small_elm_data.hdf5',
    )