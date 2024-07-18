from pathlib import Path
import dataclasses
from datetime import datetime
from typing import OrderedDict, Any, Sequence
import os
import time
import gc
import psutil

import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
import scipy.signal
# from scipy.signal import firwin, filtfilt
import h5py
import wandb

import torch
import torch.nn
import torch.cuda
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data

from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import \
    LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities import grad_norm

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
    monitor_metric: str|Any = None #'sum_loss/val' f"{task}/{metric_name}/{stage}"
    do_dropout: bool = False
    dropout_percent: float = 0.05
    use_optimizer: str = 'SGD'

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
            'precision_score': sklearn.metrics.precision_score,
            'recall_score': sklearn.metrics.recall_score,
        }

        if self.monitor_metric is None:
            self.monitor_metric = "median_classifier/f1_score/val"

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
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 4, 'kernel': (1, 4, 4), 'stride': 1},
        )

        data_shape = self.input_data_shape
        if self.is_global_zero: print(f"  Data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels: int|Any = None
        for i_layer, layer in enumerate(conv_layers):
            if i_layer != 0:
                feature_layer_dict[f"L{i_layer:02d}_bn"] = torch.nn.BatchNorm3d(
                    num_features=out_channels,
                )
            layer_name = f"L{i_layer:02d}_conv"
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

        n_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
        if self.is_global_zero: print(f"  Feature sub-model parameters: {n_params:,d}")

        return feature_model, output_size

    def make_mlp_classifier(self) -> torch.nn.Module:

        if self.is_global_zero: print("MLP classifier sub-model")

        mlp_layer_dict = OrderedDict()

        assert self.feature_space_size
        mlp_layers = (self.feature_space_size, 32, 1)

        for i in range(len(mlp_layers)-1):
            # mlp_layer_dict[f"L{i:02d}_bn"] = torch.nn.BatchNorm1d(
            #     num_features=mlp_layers[i],
            # )
            layer_name = f"L{i:02d}_fc"
            fc_layer = torch.nn.Linear(
                in_features=mlp_layers[i],
                out_features=mlp_layers[i+1],
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

        if self.is_global_zero: print(f"Using {self.use_optimizer} optimizer")
        if self.use_optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                parameter_group,
                lr=self.initial_max_lr,
                weight_decay=self.weight_decay,
                momentum=0.2,
            )
        elif self.use_optimizer.lower() == 'adam':
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

    def forward(
            self, 
            x: torch.Tensor, 
            stage: str = '',
    ) -> dict[str, torch.Tensor]:
        for layer in self.feature_model.children():
            if self.do_dropout and stage=='train':
                x = torch.nn.functional.dropout3d(x, p=self.dropout_percent)
            x = torch.nn.functional.leaky_relu(layer(x), negative_slope=self.leaky_relu_slope)
        features = x.flatten(1)
        results = {}
        for task_model_name, task_model in self.task_models.items():
            x = features
            children_layers = list(task_model.children())
            nlayers = len(children_layers)
            for i, layer in enumerate(children_layers):
                if self.do_dropout and stage=='train' and i != nlayers-1:
                    x = torch.nn.functional.dropout1d(x, p=self.dropout_percent)
                x = layer(x)
                if i != nlayers-1:
                    x = torch.nn.functional.leaky_relu(x, negative_slope=self.leaky_relu_slope)
            results[task_model_name] = x
        return results

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # if self.is_global_zero and self.global_step%50==0:
        #     print(f"  Train step {self.global_step}")
        return self.update_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='test')

    def update_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        signal_window, time_to_elm, quantiles = batch
        task_results = self(signal_window, stage=stage)
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
                    if metric_name.startswith(('f1','precision','recall')):
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
            print(f"**** Fit start with global step {self.trainer.global_step} ****")

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        if self.is_global_zero:
            print(f"Fit time: {delt/60:0.1f} min")

    def on_train_epoch_start(self):
        # if self.is_global_zero:
        #     print(f"Train epoch {self.current_epoch} start (step {self.global_step})")
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step

    def on_train_epoch_end(self):
        # if self.is_global_zero:
        #     print(f"Train epoch {self.current_epoch} end (step {self.global_step})")
        epoch_time = time.time() - self.t_train_epoch_start
        global_time = time.time() - self.t_fit_start
        epoch_steps = self.global_step-self.s_train_epoch_start
        if self.is_global_zero and self.global_step > 0:
            logged_metrics = self.trainer.logged_metrics
            line =  f"Ep {self.current_epoch:03d}  "
            line += f"train/val loss {logged_metrics['sum_loss/train']:.3f}/"
            line += f"{logged_metrics['sum_loss/val']:.3f}  "
            line += f"ep/gl steps {epoch_steps:,d}/{self.global_step:,d}  "
            line += f"ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}  " 
            print(line)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    # def on_validation_epoch_start(self):
    #     if self.is_global_zero:
    #         print(f"Validation epoch {self.current_epoch} start (step {self.global_step})")

    # def on_validation_epoch_end(self):
    #     if self.is_global_zero:
    #         print(f"Validation epoch {self.current_epoch} end (step {self.global_step})")

    def setup(self, stage=None):  # fit, validate, test, or predict
        assert self.is_global_zero == self.trainer.is_global_zero
        if self.is_global_zero:
            assert self.global_rank == 0


@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    data_file: str|Path|Any = None
    confinement_data_file: str|Path|Any = None
    max_elms: int|Any = None
    batch_size: int = 128
    stride_factor: int = 8
    num_workers: int|Any = None
    outlier_value: float = 6
    fraction_validation: float = 0.12
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = 0  # seed for ELM index shuffling; must be same across processes
    time_to_elm_quantile_min: float|Any = None
    time_to_elm_quantile_max: float|Any = None
    contrastive_learning: bool = False
    min_pre_elm_time: float|Any = None
    fir_hp_filter: float = 0.0
    bad_shots: list = None
    num_classes: int = 4
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }
    force_validation_shots: list = None
    force_test_shots: list = None
    max_shots_per_class: int = None
    max_shots: int = None
    test_only: bool = False
    n_rows: int = 8
    n_cols: int = 8
    sampling_frequency_hz: float = 1 / 10**(-6)  # Sampling frequency in Hz
    filter_taps: int = 501  # Number of taps in the filter
    lower_cutoff_frequency_hz: float = None  # Lower cutoff frequency in Hz
    upper_cutoff_frequency_hz: float = None  # Upper cutoff frequency in Hz
    clip_signals: float = None # remove signal windows with abs(raw_signals) > clip_signals
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    one_hot_labels: bool = False # if True, use one-hot vector for label
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.data_file = Path(self.data_file).absolute()
        assert self.data_file.exists()
        self.confinement_data_file = Path(self.confinement_data_file).absolute()
        assert self.confinement_data_file.exists()

        self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}
        self.global_elm_split: dict[str,Sequence] = {}
        self.global_shot_split: dict[str,np.ndarray] = {}
        self.rankwise_elm_split: dict[str,Sequence] = {}
        self.rankwise_shot_split: dict[str,Sequence] = {}
        self.rankwise_sw_split: dict[str,Sequence] = {}
        self.time_to_elm_quantiles: dict[float,float] = {}
        self.raw_signal_mean: float|Any = None
        self.raw_signal_stdev: float|Any = None

        self.confinement_datasets: dict[str,torch.utils.data.Dataset] = {}
        self.signal_mean: float|Any = None
        self.signal_stdev: float|Any = None
        self.global_confinement_split: dict[str,Sequence] = {}

        self.trainer: Trainer|Any = None
        self.batch_size_per_rank: int = 0

        if self.is_global_zero:
            print_fields(self)

        self.a_coeffs = self.b_coeffs = None
        if self.fir_hp_filter:
            self.b_coeffs = scipy.signal.firwin(
                numtaps=401,  # must be odd
                cutoff=self.fir_hp_filter,  # transition width in kHz
                pass_zero='highpass',
                fs=1e3,  # f_sample in kHz
            )
            self.a_coeffs = np.zeros_like(self.b_coeffs)
            self.a_coeffs[0] = 1

        # datamodule state, to reproduce pre-processing
        self.state_items = [
            'raw_signal_mean',
            'raw_signal_stdev',
            'global_elm_split',
            'global_confinement_split',
            'time_to_elm_quantiles',
            'signal_mean',
            'signal_stdev',
        ]
        for item in self.state_items:
            assert hasattr(self, item)

    def setup(self, stage: str):
        assert stage in ['fit', 'test', 'predict']
        assert self.is_global_zero == self.trainer.is_global_zero

        assert self.batch_size % self.trainer.world_size == 0
        self.batch_size_per_rank = self.batch_size // self.trainer.world_size
        if self.is_global_zero:
            print(f"Batch size: {self.batch_size}")
            print(f"Batch size per rank {self.batch_size_per_rank}")

        if 'train' in self.global_elm_split and len(self.global_elm_split['train'])>0:
            if self.is_global_zero:
                print("Reusing saved global data split")
        else:
            if self.is_global_zero:
                print("Creating global data split")
            self._make_data_split()

        if 'train' not in self.global_confinement_split:
            if self.is_global_zero:
                print("Creating global confinement split")
            self._get_confinement_events_and_split()
        self.dataset_confinement_events = {
                'train': self.train_confinement_events,
                'validation': self.validation_confinement_events,
                'test': self.test_confinement_events,
                'predict': self.test_confinement_events,
            }

        if self.is_global_zero:
            if self.b_coeffs is not None:
                print(f"  Using HP filter with f_pass={self.fir_hp_filter:.1f} kHz")
            else:
                print("  Using raw BES signals; no HP filter")

        stages = ['train', 'validation'] if stage == 'fit' else [stage]
        for st in stages:
            assert st in ['train', 'validation', 'test', 'predict']
            if st in self.elm_datasets and isinstance(self.elm_datasets[st], torch.utils.data.Dataset):
                if self.is_global_zero:
                    print(f"Stage {st.upper()}: Using saved dataset")
                continue
            print(f"Rank {self.trainer.global_rank} Stage {st.upper()}: data setup")
            global_elm_indices = self.global_elm_split[st]
            if self.is_global_zero: 
                print(f"  Global ELM count: {len(global_elm_indices)}")
            assert len(global_elm_indices) > 0
            global_sw_metadata_list = []
            global_outliers = 0
            skipped_short_pre_elm_time = 0
            with h5py.File(self.data_file, 'r') as h5_file:
                elms: h5py.Group = h5_file['elms']
                for i_elm, elm_index in enumerate(global_elm_indices):
                    if i_elm%100 == 0 and self.is_global_zero:
                        print(f"  Reading ELM event {i_elm:04d}/{len(global_elm_indices):04d}")
                    elm_event: h5py.Group = elms[f"{elm_index:06d}"]
                    shot = int(elm_event.attrs['shot'])
                    assert elm_event["bes_signals"].shape[0] == 64
                    assert elm_event['bes_time'].size == elm_event["bes_signals"].shape[1]
                    time = np.array(elm_event['bes_time'], dtype=np.float32)
                    t_start: float = elm_event.attrs['t_start']
                    t_stop: float = elm_event.attrs['t_stop'] - 0.05
                    if self.min_pre_elm_time and (t_stop-t_start) < self.min_pre_elm_time:
                        skipped_short_pre_elm_time += 1
                        continue
                    i_start: int = np.flatnonzero(time >= t_start)[0]
                    i_stop: int = np.flatnonzero(time <= t_stop)[-1]
                    i_window_stop = i_stop
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    if self.b_coeffs is not None:
                        signals = np.array(
                            scipy.signal.lfilter(
                                x=signals,
                                a=self.a_coeffs,
                                b=self.b_coeffs,
                            ),
                            dtype=np.float32,
                        )
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                    assert signals.shape[0] == time.size
                    assert (signals.shape[1] == 8) and (signals.shape[2] == 8)
                    while True:
                        i_window_start = i_window_stop - self.signal_window_size
                        if i_window_start < i_start: break
                        if self.outlier_value:
                            signal_window = signals[i_window_start:i_window_stop, ...]
                            assert signal_window.shape[0] == self.signal_window_size
                            if np.abs(signal_window).max() > self.outlier_value:
                                i_window_stop -= self.signal_window_size // self.stride_factor
                                global_outliers += 1
                                continue
                        global_sw_metadata_list.append({
                            'elm_index': elm_index,
                            'shot': shot,
                            'i_t0': i_window_start,
                            'time_to_elm': time[i_stop] - time[i_window_stop]
                        })
                        i_window_stop -= self.signal_window_size // self.stride_factor

            print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}:  Skipped ELMs for short pre-ELM time: {skipped_short_pre_elm_time}")

            n_signal_windows = len(global_sw_metadata_list)
            print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}: Global signal windows: {n_signal_windows:,d}  ({global_outliers:,d} outliers removed)")
            print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}: Global steps per epoch {n_signal_windows/self.batch_size:,.1f}")

            # Raw signal stats
            self._get_statistics(signal_windows=global_sw_metadata_list, stage=st)
            assert self.raw_signal_mean and self.raw_signal_stdev
            if st == 'train':
                self.save_hyperparameters({
                    'raw_signal_mean': self.raw_signal_mean,
                    'raw_signal_stdev': self.raw_signal_stdev,
                })

            # time-to-ELM quantiles
            if st == 'train':
                if self.is_global_zero:
                    print("  Calculating time-to-ELM quantiles")
                quantiles = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
                time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in global_sw_metadata_list]
                quantile_values = np.quantile(time_to_elm_labels, quantiles)
                self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
                self.save_hyperparameters({'time_to_elm_quantiles': self.time_to_elm_quantiles})
                if self.is_global_zero: 
                    print(f"  Time-to-ELM quantiles for binary labels:")
                    for q, qval in self.time_to_elm_quantiles.items():
                        print(f"    Quantile {q:.2f}: {qval:.1f} ms")
            assert self.time_to_elm_quantiles

            # restrict data according to quantiles
            if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in global_sw_metadata_list])
                time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                if self.contrastive_learning:
                    if self.is_global_zero: 
                        print(f"  Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                    for i in np.arange(len(global_sw_metadata_list)-1, -1, -1, dtype=int):
                        if (global_sw_metadata_list[i]['time_to_elm'] > time_to_elm_min) and \
                            (global_sw_metadata_list[i]['time_to_elm'] < time_to_elm_max):
                            global_sw_metadata_list.pop(i)
                else:
                    if self.is_global_zero: 
                        print(f"  Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                    for i in np.arange(len(global_sw_metadata_list)-1, -1, -1, dtype=int):
                        if (global_sw_metadata_list[i]['time_to_elm'] < time_to_elm_min) or \
                            (global_sw_metadata_list[i]['time_to_elm'] > time_to_elm_max):
                            global_sw_metadata_list.pop(i)
                if self.is_global_zero:
                    n_signal_windows = len(global_sw_metadata_list)
                    print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}: Restricted global signal windows: {n_signal_windows:,d}")
                    print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}: Global steps per epoch {n_signal_windows/self.batch_size:,.1f}")

            # split signal windows by rank
            rankwise_sw_split = np.array_split(global_sw_metadata_list, self.trainer.world_size)
            sw_for_rank = list(rankwise_sw_split[self.trainer.global_rank])
            elms_for_rank = np.unique(np.array(
                [item['elm_index'] for item in sw_for_rank],
                dtype=int,
            ))
            shots_for_rank = np.unique(np.array(
                [item['shot'] for item in sw_for_rank],
                dtype=int,
            ))
            print(f"  Rank {self.trainer.global_rank} Stage {st.upper()}:  Shots/ELMs/SigWin: {len(shots_for_rank):,d}/{len(elms_for_rank):,d}/{len(sw_for_rank):,d}")

            # get rank-wise ELM signals
            signals_for_rank = {}
            with h5py.File(self.data_file) as root:
                for elm_index in elms_for_rank:
                    elm_group: h5py.Group = root['elms'][f"{elm_index:06d}"]
                    signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals).reshape(1, -1, 8, 8)  # reshape to (time, pol, rad)
                    # normalized signals
                    signals_for_rank[elm_index] = (signals - self.raw_signal_mean) / self.raw_signal_stdev
            assert len(signals_for_rank) == len(elms_for_rank)

            # rank-wise datasets
            if st in ['train', 'validation', 'test']:
                self.elm_datasets[st] = ELM_TrainValTest_Dataset(
                    signal_window_size=self.signal_window_size,
                    time_to_elm_quantiles=self.time_to_elm_quantiles,
                    sw_list=sw_for_rank,
                    signal_list=signals_for_rank,
                    quantile_min=self.time_to_elm_quantile_min,
                    quantile_max=self.time_to_elm_quantile_max,
                    contrastive_learning=self.contrastive_learning,
                )
                print(f"  Rank {self.trainer.global_rank} stage {st}: Dataset size: {len(self.elm_datasets[st]):,d}")
            
            if st in ['test', 'predict']:
                pass

        def get_time_for_index(shot_event_tuple):
            shot, event = shot_event_tuple  # Unpack the tuple
            with h5py.File(self.confinement_data_file) as h5_file:
                event_key = f"{shot}/{event}"  # Updated to use shot/event structure
                time_count = h5_file[event_key]["signals"].shape[1]
            return time_count

        for dataset_stage in stages:
            # Determine the chunk of confinement indices for this GPU
            events = self.dataset_confinement_events[dataset_stage]
            times = [get_time_for_index(shot_event) for shot_event in events]  # Adapted for (shot, event) tuples
            if dataset_stage in ['train']:
                print(f"Creating chunks for {dataset_stage} with {len(events)} indices and total time {sum(times)}")
                # Create balanced chunks
                # chunks = self.create_balanced_chunks(events, times, self.trainer.world_size)
                # Create a mapping from indices to times
                index_to_time = {index: time for index, time in zip(events, times)}
                # Create a list to hold the chunks, and a list to hold the total time for each chunk
                chunks = [[] for _ in range(self.trainer.world_size)]
                chunk_times = [0] * self.trainer.world_size
                # Iterate over the indices, sorted by time from largest to smallest
                for index, time in sorted(index_to_time.items(), key=lambda item: item[1], reverse=True):
                    # Find the chunk with the shortest total time so far
                    min_time_chunk_idx = min(range(self.trainer.world_size), key=lambda i: chunk_times[i])
                    # Add this index to that chunk
                    chunks[min_time_chunk_idx].append(index)
                    # Update the total time for that chunk
                    chunk_times[min_time_chunk_idx] += time
                # Print information about the chunks
                for i, (chunk, chunk_time) in enumerate(zip(chunks, chunk_times)):
                    print(f"Chunk {i} size: {len(chunk)}, total time: {sum(index_to_time[index] for index in chunk)}")
                # return chunks
                # Determine the chunk for this GPU
                chunk_events = chunks[self.trainer.global_rank]
            elif dataset_stage in ['validation', 'test', 'predict']:
                chunk_events = events

            dataset = self._load_and_preprocess_data_2(chunk_events, dataset_stage)

            # Store the DataLoader for this GPU
            if dataset_stage == 'train':
                self._train_dataloader = torch.utils.data.DataLoader(
                                        dataset, 
                                        batch_size=self.batch_size,
                                        shuffle=True,             
                                        num_workers=self.num_workers,
                                        persistent_workers=(self.num_workers > 0),
                                        drop_last=True,
                                        )

    def _load_and_preprocess_data_2(self, shot_event_indices, dataset_stage):
        # t0 = time.time()
        print(f"Reading confinement events for dataset `{dataset_stage}`")
        confinement_data = []
        # n_bins = 201
        # cummulative_hist = np.zeros(n_bins, dtype=int)
        # selected_channels = self.n_rows * self.n_cols  # Total number of channels to select

        with h5py.File(self.confinement_data_file, 'r') as h5_file:
            if len(shot_event_indices) >= 5:
                print(f"  Initial shot/event indices: {shot_event_indices[:5]}")
            time_counts = []
            long_enough_indices = []  # List to hold indices of events with long enough signals
            for i, (shot, event) in enumerate(shot_event_indices):
                event_key = f"{shot}/{event}"
                signal_length = h5_file[event_key]["signals"].shape[1]
                
                # Check if the signal length is greater than or equal to self.signal_window_size
                if signal_length >= self.signal_window_size:
                    inboard_order = h5_file[shot].attrs.get("inboard_column_channel_order", None)

                    # Skip processing if inboard_order is missing or empty
                    if inboard_order is None or len(inboard_order) == 0:
                        print(f"Skipping event {event_key} due to missing or empty inboard_column_channel_order.")
                        continue
                    
                    time_counts.append(signal_length)
                    long_enough_indices.append((shot, event))

            time_count = np.sum(time_counts)
            discarded_count = len(shot_event_indices) - len(long_enough_indices)
            print(f"Discarded {discarded_count} events due to insufficient signal length or missing inboard order.")
            
            packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
            start_index = 0

            if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                bandpass_filter = scipy.signal.firwin(
                    self.filter_taps,
                    [self.lower_cutoff_frequency_hz, self.upper_cutoff_frequency_hz],
                    pass_zero=False,
                    fs=self.sampling_frequency_hz
                )
            else:
                bandpass_filter = None

            for i, (shot, event) in enumerate(long_enough_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(shot_event_indices):04d} in shot {shot}")
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Retrieve signals and reshape according to inboard_order
                signals = np.array(event_data["signals"][:, :], dtype=np.float32)
                signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                if bandpass_filter is not None and signals.shape[0] > 3 * self.filter_taps:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = scipy.signal.filtfilt(bandpass_filter, 1, signals, axis=0)
                labels = np.array(event_data["labels"], dtype=int)

                # labels, valid_t0 = self._get_valid_indices(labels)
                valid_t0 = np.zeros(labels.size, dtype=int)
                first_valid_signal_window_start_index = self.signal_window_size - 1
                valid_t0[first_valid_signal_window_start_index:] = 1
                packaged_signals[start_index:start_index + signals.shape[0]] = signals
                start_index += signals.shape[0]
                confinement_data.append({
                    'labels': labels, 
                    'valid_t0': valid_t0,
                    'confinement_mode_key': event_key,
                    'shot': shot,
                    'time': event,
                })

        print(f"  Global min/max raw signal, ch 1-32: {np.amin(packaged_signals[:,:4,:]):.6f}, {np.amax(packaged_signals[:,:4,:]):.6f}")
        print(f"  Global min/max raw signal, ch 33-64: {np.amin(packaged_signals[:,4:,:]):.6f}, {np.amax(packaged_signals[:,4:,:]):.6f}")

        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in confinement_data], axis=0)
        if self.one_hot_labels:
            encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(self.num_classes)], handle_unknown='ignore')
            packaged_labels = encoder.fit_transform(packaged_labels.reshape(-1, 1))

        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in confinement_data], axis=0)

        # start indices for each confinement mode event in concatenated dataset
        packaged_window_start = []
        index = 0
        for confinement_mode in confinement_data:
            packaged_window_start.append(index)
            index += confinement_mode['labels'].size
        packaged_window_start = np.array(packaged_window_start, dtype=int)

        packaged_confinement_mode_key = np.array(
            [confinement_mode['confinement_mode_key'] for confinement_mode in confinement_data],
            dtype=str,
        )
        del confinement_data

        # valid t0 indices
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        print("  Raw data stats")
        stats = self._get_statistics2(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

        # mask abs(signals) > N volts
        if self.clip_signals and dataset_stage == 'train':
            print(f"  Clipping signal windows beyond +/- {self.clip_signals} V")
            mask = []
            for i in packaged_valid_t0_indices:
                signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                mask.append((signal_window.min() >= -self.clip_signals) and (signal_window.max() <= self.clip_signals))
            packaged_valid_t0_indices = packaged_valid_t0_indices[mask]

            stats = self._get_statistics2(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            print(f"  Clipped signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        # mask outlier signals
        if self.mask_sigma_outliers:
            if None in [self.mask_lb, self.mask_ub]:
                assert dataset_stage == 'train' or not self.train_confinement_events, f"Dataset_stage: {dataset_stage}"
                print(f"  Calculating mask upper/lower bounds from {dataset_stage} data")
                self.mask_lb = stats['mean'] - self.mask_sigma_outliers * stats['stdev']
                self.mask_ub = stats['mean'] + self.mask_sigma_outliers * stats['stdev']
                self.save_hyperparameters({
                    'mask_lb': self.mask_lb.item(),
                    'mask_ub': self.mask_ub.item(),
                })
            print(f"  Mask {self.mask_sigma_outliers:.2f} sigma outliers from signals")
            print(f"  Mask lower bound {self.mask_lb:.3f} upper bound {self.mask_ub:.3f}")
            mask = np.zeros(packaged_valid_t0_indices.size, dtype=bool)
            for i_t0_index, t0_index in enumerate(packaged_valid_t0_indices):
                signal_window = packaged_signals[t0_index: t0_index + self.signal_window_size, :, :]
                mask[i_t0_index] = (
                    np.max(signal_window) <= self.mask_ub and
                    np.min(signal_window) >= self.mask_lb
                )
            packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
            print("  Masked data stats")
            stats = self._get_statistics2(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
        
        # standardize signals based on training data
        if None in [self.signal_mean, self.signal_stdev]:
            assert dataset_stage == 'train' or not self.train_confinement_events, f"Dataset_stage: {dataset_stage}"
            print(f"  Calculating signal mean and std from {dataset_stage} data")
            self.signal_mean = stats['mean']
            self.signal_stdev = stats['stdev']
            self.signal_exkurt = stats['exkurt']
            self.save_hyperparameters({
                'signal_mean': self.signal_mean.item(),
                'signal_stdev': self.signal_stdev.item(),
                'signal_exkurt': self.signal_exkurt.item(),
            })

        if dataset_stage in ['train']:
            print(f"  Standarizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f}")
            print(f"  Standardized signal stats")
            for idx, signal in enumerate(packaged_signals):
                packaged_signals[idx] = (signal - self.signal_mean) / self.signal_stdev
            stats = self._get_statistics2(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
        self.max_abs_valid_signal = np.max(np.abs([stats['min'],stats['max']]))
            
        if dataset_stage in ['train']:
            dataset = Confinement_TrainValTest_Dataset(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                window_start_indices=packaged_window_start,
                signal_window_size=self.signal_window_size,
                confinement_mode_keys=packaged_confinement_mode_key,
            )
            return dataset
        if dataset_stage in ['validation', 'test']:
            self.confinement_datasets[dataset_stage] = Confinement_TrainValTest_Dataset(
                    signals=packaged_signals,
                    n_rows=self.n_rows,
                    n_cols=self.n_cols,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    window_start_indices=packaged_window_start,
                    signal_window_size=self.signal_window_size,
                    confinement_mode_keys=packaged_confinement_mode_key,
                )
            return
        if dataset_stage in ['predict']:
            del self._train_dataloader
            del self.confinement_datasets['validation']
            
            # predict_datasets = []
            # for i_confinement_mode, idx_start in enumerate(packaged_window_start):
            #     if self.max_predict_confinement_modes and i_confinement_mode == self.max_predict_confinement_modes:
            #         break
            #     if i_confinement_mode == packaged_window_start.size - 1:
            #         idx_stop = packaged_labels.size - 1
            #     else:
            #         idx_stop = packaged_window_start[i_confinement_mode+1]-1
            #     dataset = Confinement_Predict_Dataset(
            #         signals=packaged_signals[idx_start:idx_stop, ...],
            #         labels=packaged_labels[idx_start:idx_stop],
            #         signal_window_size=self.signal_window_size,
            #         shot=packaged_shot[i_confinement_mode],
            #         start_time=packaged_start_time[i_confinement_mode],
            #         confinement_mode_index=packaged_confinement_mode_key[i_confinement_mode],
            #     )
            #     predict_datasets.append(dataset)
            # self.confinement_datasets['predict'] = predict_datasets
            # return predict_datasets
        # print(f"  Data stage `{dataset_stage}` elapsed time {(time.time()-t0)/60:.1f} min")
        gc.collect()
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _get_statistics2(
            self, 
            sample_indices: np.ndarray, 
            signals: np.ndarray,
    ) -> dict:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_samples = int(100e3)
        stat_interval = np.max([1, sample_indices.size//stat_samples])
        n_samples = sample_indices.size // stat_interval
        for i in sample_indices[::stat_interval]:
            signal_window = signals[i: i + self.signal_window_size, :, :]
            signal_min = np.min([signal_min, signal_window.min()])
            signal_max = np.max([signal_max, signal_window.max()])
            hist, bin_edges = np.histogram(
                signal_window,
                bins=n_bins,
                range=[-10.4, 10.4],
            )
            cummulative_hist += hist
        bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
        stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
        exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
        print(f"    Stats: count {sample_indices.size:,} min {signal_min:.3f} max {signal_max:.3f} mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} n_samples {n_samples:,}")
        return {
            'count': sample_indices.size,
            'min': signal_min,
            'max': signal_max,
            'mean': mean,
            'stdev': stdev,
            'exkurt': exkurt,
        }

    def _get_confinement_events_and_split(self):
        print(f"Data file: {self.confinement_data_file}")
        if self.bad_shots is None:
            self.bad_shots = []  # Initialize to empty list if None
        shots = {}
        with h5py.File(self.confinement_data_file, "r") as data_file:
            for shot in data_file.keys():
                if shot in self.bad_shots:
                    print(f"Skipping bad shot: {shot}")
                    continue
                shot_labels = []
                for event in data_file[shot].keys():
                    event_data = data_file[shot][event]
                    if 'labels' in event_data:
                        labels_array = event_data['labels'][()]
                        shot_labels.extend(labels_array)
                if not shot_labels:
                    continue
                label_presence = tuple(class_id in shot_labels for class_id in range(self.num_classes))  
                shot_events = [(shot, event) for event in data_file[shot].keys() if 'labels' in data_file[shot][event]]
                # metadata = self._extract_metadata(data_file[shot].attrs)
                attrs = data_file[shot].attrs
                metadata = {
                    'r_avg': attrs.get('r_avg'),
                    'z_avg': attrs.get('z_avg'),
                    'delz_avg': attrs.get('delz_avg')
                }
                shots[shot] = (shot_events, label_presence, metadata)
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        def _check_bounds(value, bounds):
            return bounds[0] <= value <= bounds[1] if bounds else True
        for shot in list(shots):
            metadata = shots[shot][2]
            # if not self._metadata_within_bounds(metadata):
            if not all(_check_bounds(metadata[key], self.metadata_bounds[key]) for key in ['r_avg', 'z_avg', 'delz_avg'] if key in self.metadata_bounds):
                shots.pop(shot)
                if metadata['r_avg'] is None or not _check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
                    r_avg_exclusions += 1
                if metadata['z_avg'] is None or not _check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
                    z_avg_exclusions += 1
                if metadata['delz_avg'] is None or not _check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
                    delz_avg_exclusions += 1
        print(f"Number of r_avg exclusions: {r_avg_exclusions}")
        print(f"Number of z_avg exclusions: {z_avg_exclusions}")
        print(f"Number of delz_avg exclusions: {delz_avg_exclusions}")
        test_shot_data = {}
        validation_shot_data = {}
        # Handling forced test shots
        if self.force_test_shots:
            for shot_number in self.force_test_shots:
                if shot_number in shots:
                    test_shot_data[shot_number] = shots.pop(shot_number)
                else:
                    print(f"Warning: Forced test shot number {shot_number} not found in dataset.")
        # Handling forced validation shots
        if self.force_validation_shots:
            for shot_number in self.force_validation_shots:
                if shot_number in shots:
                    validation_shot_data[shot_number] = shots.pop(shot_number)
                else:
                    print(f"Warning: Forced validation shot number {shot_number} not found in dataset.")
        # These dictionaries can be used to ensure that the specified shots are included in their respective datasets
        self.forced_test_shots_data = test_shot_data
        self.forced_validation_shots_data = validation_shot_data
        shots_by_class = {}
        for shot, (events, labels, metadata) in shots.items():
            if labels not in shots_by_class:
                shots_by_class[labels] = []
            shots_by_class[labels].append(shot)
        if self.max_shots_per_class is not None:
            for labels, shot_list in shots_by_class.items():
                if len(shot_list) > self.max_shots_per_class:
                    shots_by_class[labels] = np.random.choice(shot_list, self.max_shots_per_class, replace=False).tolist()
        filtered_shots =  {shot: shots[shot] for label_shots in shots_by_class.values() for shot in label_shots}
        shot_numbers = np.array(list(filtered_shots.keys()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(shot_numbers)
        if self.max_shots:
            shot_numbers = shot_numbers[:self.max_shots]
        self.all_confinement_events = np.concatenate([filtered_shots[shot][0] for shot in shot_numbers])
        if not self.test_only:
            shot_numbers = np.array(list(filtered_shots.keys()))
            # Map labels here
            # labels = [self.map_labels(filtered_shots[shot][1]) for shot in shot_numbers]
            labels = [filtered_shots[shot][1] for shot in shot_numbers]
            # Try to stratify, revert to random split if stratification is not possible
            try:
                train_indices, test_val_indices = train_test_split(shot_numbers, labels, test_size=self.fraction_test + self.fraction_validation, stratify=labels, random_state=self.seed)
            except ValueError:
                print("Stratified split failed; reverting to random split for train/test+validation sets.")
                train_indices, test_val_indices = train_test_split(shot_numbers, test_size=self.fraction_test + self.fraction_validation, random_state=self.seed)
            try:
                test_indices, val_indices = train_test_split(
                    test_val_indices,
                    [filtered_shots[shot][1] for shot in test_val_indices],
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    # stratify=[self.map_labels(filtered_shots[shot][1]) for shot in test_val_indices],
                    stratify=[filtered_shots[shot][1] for shot in test_val_indices],
                    random_state=self.seed
                )
            except ValueError:
                print("Stratified split failed; reverting to random split for test/validation sets.")
                test_indices, val_indices = train_test_split(
                    test_val_indices,
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    random_state=self.seed  # No stratification here
                )

            # Ensure forced shots are included back in filtered_shots if needed
            filtered_shots.update(self.forced_test_shots_data)
            filtered_shots.update(self.forced_validation_shots_data)

            # Include forced test and validation shots
            if hasattr(self, 'forced_test_shots_data'):
                forced_test_indices = np.array(list(self.forced_test_shots_data.keys()))
                test_indices = np.concatenate((test_indices, forced_test_indices))
            
            if hasattr(self, 'forced_validation_shots_data'):
                forced_val_indices = np.array(list(self.forced_validation_shots_data.keys()))
                val_indices = np.concatenate((val_indices, forced_val_indices))

            # Assign events to datasets
            self.train_confinement_events = [event for shot in train_indices for event in filtered_shots[shot][0]]
            self.validation_confinement_events = [event for shot in val_indices for event in filtered_shots[shot][0]]
            self.test_confinement_events = [event for shot in test_indices for event in filtered_shots[shot][0]]

            print(f"Train set size: {len(self.train_confinement_events)} events")
            print(f"Validation set size: {len(self.validation_confinement_events)} events")
            print(f"Test set size: {len(self.test_confinement_events)} events")

            print(f"Train shot numbers: {train_indices}")
            print(f"Validation shot numbers: {val_indices}")
            print(f"Test shot numbers: {test_indices}")
        else:
            shot_numbers = np.array(list(filtered_shots.keys()))
            self.test_confinement_events = [event for shot in shot_numbers for event in filtered_shots[shot][0]]
        mode_times = {}  # Time spent in each mode
        mode_shots = {}  # Unique shots for each mode

        with h5py.File(self.confinement_data_file,) as file:
            for shot_key in list(filtered_shots.keys()):
                if shot_key in file:
                    shot_group = file[shot_key]
                    for nested_key in shot_group.keys():
                        nested_group = shot_group[nested_key]
                        if 'labels' in nested_group and 'time' in nested_group:
                            labels = nested_group['labels'][:]
                            time = nested_group['time'][:]
                            time_diffs = np.diff(time)  # Time intervals

                            # Handle NaNs in labels: create a mask for valid (non-NaN) labels
                            valid_indices = ~np.isnan(labels[:-1])  # Exclude the last label as it has no following time difference

                            # Iterate only over valid indices
                            for i in np.where(valid_indices)[0]:  # Get the indices of valid labels
                                label = int(labels[i])  # Convert to int, NaNs should have been filtered out
                                if label not in mode_times:
                                    mode_times[label] = 0  # Initialize if the label hasn't been encountered yet
                                    mode_shots[label] = set()  # Initialize a set for unique shots

                                mode_times[label] += time_diffs[i]  # Add time difference to the corresponding mode
                                mode_shots[label].add(shot_key)  # Add shot key to the set of unique shots for the mode

        # Convert time from microseconds to more suitable units if necessary
        mode_times_seconds = {k: v / 1e3 for k, v in mode_times.items()}  # Convert to seconds

        # Count unique shots for each mode
        mode_shot_counts = {k: len(v) for k, v in mode_shots.items()}

        print("Total Time Spent in Each Mode (seconds):", mode_times_seconds)
        print("Number of Unique Shots for Each Mode:", mode_shot_counts)

    # def _get_valid_indices(
    #     self,
    #     labels: np.ndarray = None,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     # Determine valid t0 indices (start of signal windows) for real-time inference
    #     valid_t0 = np.zeros(labels.size, dtype=int)
    #     first_valid_signal_window_start_index = self.signal_window_size - 1
    #     valid_t0[first_valid_signal_window_start_index:] = 1

    #     # if self.log_time:
    #     #     labels = np.log10(labels)

    #     return labels, valid_t0
    
    # def apply_bandpass_filter(self, signals):
    #         """
    #         Applies a bandpass filter to the given signals if the cutoff frequencies are specified.
    #         Otherwise, returns the original signals.

    #         Args:
    #             packaged_signals: The signals to be filtered.

    #         Returns:
    #             Filtered signals or the original signals.
    #         """
    #         required_length = 3 * self.filter_taps  # Set to 3 times the number of filter taps

    #         # Check if the cutoff frequencies are specified
    #         if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None and signals.shape[0] > required_length:
    #             # Design the bandpass filter
    #             bandpass_filter = scipy.signal.firwin(
    #                 self.filter_taps,
    #                 [self.lower_cutoff_frequency_hz, self.upper_cutoff_frequency_hz],
    #                 pass_zero=False,
    #                 fs=self.sampling_frequency_hz
    #             )

    #             # Apply the filter
    #             filtered_signals = scipy.signal.filtfilt(bandpass_filter, 1, signals, axis=0)
    #             return filtered_signals
    #         else:
    #             # print("BANDPASS FILTER NOT APPLIED")
    #             return signals

    def _make_data_split(self):
        assert len(self.global_elm_split) == 0
        print(f"Rank {self.trainer.global_rank}: Data split")
        with h5py.File(self.data_file, 'r') as root:
            global_shots = set([int(shot_key) for shot_key in root['shots']])
            shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(global_shots ^ shots_from_elms) == 0
            global_elms = [int(elm_key) for elm_key in root['elms']]
            global_shots = list(global_shots)
            if self.is_global_zero: 
                print(f"  ELMs/shots in HDF5 file: {len(global_elms):,d} / {len(global_shots):,d}")
            # limit max ELMs
            if self.max_elms and len(global_elms) > self.max_elms:
                global_elms = global_elms[:self.max_elms]
                global_shots = set([int(root['elms'][f"{elm_index:06d}"].attrs['shot']) for elm_index in global_elms])
                global_shots = list(global_shots)
                if self.is_global_zero:
                    print(f"  ELMs/shots for analysis: {len(global_elms):,d} / {len(global_shots):,d}")
            # shuffle shots in dataset
            print(f"  Rank {self.trainer.global_rank}: Shuffling global shots with seed={self.seed}")
            np.random.default_rng(self.seed).shuffle(global_shots)
            print(f"  Rank {self.trainer.global_rank}: Shuffled shot order: " + ', '.join(map(str, global_shots[:5])))
            # order ELMs by shuffled shots
            print(f"  Rank {self.trainer.global_rank}: Ordering ELMs by shuffled shots")
            new_global_elms = []
            for shot in global_shots:
                for elm_key in root['elms']:
                    if root['elms'][elm_key].attrs['shot'] == shot:
                        new_global_elms.append(int(elm_key))
            global_elms = new_global_elms
            # split shots
            n_test_shots = int(self.fraction_test * len(global_shots))
            n_validation_shots = int(self.fraction_validation * len(global_shots))
            self.global_shot_split['test'], self.global_shot_split['validation'], self.global_shot_split['train'] = \
                np.split(global_shots, [n_test_shots, n_test_shots+n_validation_shots])
            
            for stage in ['train','validation','test']:
                self.global_elm_split[stage] = [
                    int(key) 
                    for key, value in root['elms'].items()
                    if value.attrs['shot'] in self.global_shot_split[stage]
                ]
                print(f"  Rank {self.trainer.global_rank} Stage {stage.upper()}: Global ELM/shot count {len(self.global_elm_split[stage]):,d} ({len(self.global_elm_split[stage])/len(global_elms)*1e2:.1f}%) / {self.global_shot_split[stage].size} ({self.global_shot_split[stage].size/len(global_shots)*1e2:.1f}%)")
                # self.rankwise_shot_split[stage] = np.array_split(shot_split, self.trainer.world_size)
                # self.rankwise_elm_split[stage] = [
                #     [
                #         int(key) 
                #         for key, value in root['elms'].items() 
                #         if value.attrs['shot'] in rank_shot_list
                #     ]
                #     for rank_shot_list in self.rankwise_shot_split[stage]
                # ]
                # print(f"  Rank {self.trainer.global_rank} stage {stage.upper()}: Rank ELM/shot count {len(self.rankwise_elm_split[stage][self.trainer.global_rank])} / {self.rankwise_shot_split[stage][self.trainer.global_rank].size}")

        # if self.is_global_zero: 
        #     print("ELMs for analysis")
        # for stage, elm_indices in self.elm_split.items():
        #     if self.is_global_zero: 
        #         print(f"  {stage} ELMs: {len(elm_indices)}  ({len(elm_indices)/len(elms)*1e2:.1f}%)")

    def _get_statistics(
            self, 
            signal_windows: list[dict],
            stage: str,
    ):
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
                    elm_event: h5py.Group = root['elms'][f'{elm_index:06d}']
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

    def _train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        # is_distributed = self.trainer.world_size > 1
        # shuffle = True if stage=='train' else False
        # sampler = torch.utils.data.DistributedSampler(
        #     dataset=self.datasets[stage],
        #     shuffle=shuffle,
        # ) if is_distributed else None
        sampler = (
            torch.utils.data.RandomSampler(data_source=self.elm_datasets[stage])
            if stage == 'train'
            else torch.utils.data.SequentialSampler(data_source=self.elm_datasets[stage])
        )
        if self.num_workers is None:
            self.num_workers = 2 if self.trainer.world_size==1 else 0
        return torch.utils.data.DataLoader(
            dataset=self.elm_datasets[stage],
            sampler=sampler,
            batch_size=self.batch_size_per_rank,  # batch size per rank
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers else None,
            pin_memory=True,
            # persistent_workers=bool(self.num_workers),
            drop_last=True,
        )


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signal_window_size: int = 0
    sw_list: list|Any = None # global signal window data mapping to dataset index
    signal_list: dict|Any = None # rank-wise signals (map to ELM indices)
    time_to_elm_quantiles: dict[float, float]|Any = None
    quantile_min: float|Any = None
    quantile_max: float|Any = None
    contrastive_learning: bool = False

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        for elm_index in self.signal_list:
            self.signal_list[elm_index] = torch.from_numpy(
                self.signal_list[elm_index]
            )

    def __len__(self) -> int:
        return len(self.sw_list)
    
    def __getitem__(self, i: int) -> tuple:
        sw_metadata = self.sw_list[i]
        i_t0 = sw_metadata['i_t0']
        time_to_elm = sw_metadata['time_to_elm']
        elm_index = sw_metadata['elm_index']
        signals = self.signal_list[elm_index]
        signal_window = signals[..., i_t0 : i_t0 + self.signal_window_size, :, :]
        quantile_binary_label = {q: int(time_to_elm<=qval) for q, qval in self.time_to_elm_quantiles.items()}
        return signal_window, time_to_elm, quantile_binary_label


class Confinement_TrainValTest_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            n_rows: int,
            n_cols: int,
            labels: np.ndarray,
            sample_indices: np.ndarray,
            window_start_indices: np.ndarray,
            signal_window_size: int,
            confinement_mode_keys: np.ndarray,
    ) -> None:
        # Create a contiguous copy of the array and then convert it to a PyTorch tensor
        self.signals = torch.from_numpy(np.ascontiguousarray(signals)[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == n_rows and
            self.signals.size(3) == n_cols
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        # assert self.labels.ndim == 1, "Labels have incorrect shape"
        print(signals.shape, labels.shape)
        # assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.signal_window_size = signal_window_size
        self.window_start_indices = torch.from_numpy(window_start_indices)
        self.sample_indices = torch.from_numpy(sample_indices)
        self.confinement_mode_keys = confinement_mode_keys
        assert torch.max(self.sample_indices) < self.labels.shape[0]


        # Create a dictionary to map confinement_mode_keys to start and end indices
        self.confinement_mode_id_to_indices = {}
        for idx, key in enumerate(self.confinement_mode_keys):
            start_idx = self.window_start_indices[idx]
            end_idx = self.window_start_indices[idx + 1] if idx + 1 < len(self.window_start_indices) else self.signals.size(1)
            self.confinement_mode_id_to_indices[key] = (start_idx, end_idx)

    def __len__(self) -> int:
        return self.sample_indices.numel()
    
    def get_full_signal_by_id(self, confinement_mode_id):    
        # Make sure to check what type of key is stored in confinement_mode_id_to_indices
        start_idx, end_idx = self.confinement_mode_id_to_indices.get(confinement_mode_id, (None, None))

        if start_idx is None or end_idx is None:
            print(f"Warning: No indices found for confinement_mode_id: {confinement_mode_id}")
            return None
        if self.n_rows >= 3:
            row_idx = 2
        else:
            row_idx = self.n_rows-1

        if self.n_cols >= 4:
            col_idx = 3
        else:
            col_idx = self.n_cols-1

        print(f"Found start index {start_idx}, end index {end_idx} for confinement_mode_id: {confinement_mode_id}")
        return self.signals[:, start_idx:end_idx, row_idx, col_idx].squeeze(0)

    def __getitem__(self, i: int) -> tuple:
        # Retrieve the index from sample_indices that is guaranteed to have enough previous data
        i_t0 = self.sample_indices[i]

        # Define the start index for the signal window to look backwards
        start_index = i_t0 - self.signal_window_size + 1

        # Retrieve the signal window from start_index to i_t0 (inclusive)
        signal_window = self.signals[:, start_index:i_t0 + 1, :, :]

        # The label is typically the current index in real-time scenarios
        label = self.labels[i_t0: i_t0 + 1]

        # Look up the correct confinement_mode_key based on i_t0
        confinement_mode_idx = (self.window_start_indices <= i_t0).nonzero().max()
        confinement_mode_key = self.confinement_mode_keys[confinement_mode_idx]

        # Convert the key to an integer by removing non-numeric characters and converting to int
        confinement_mode_id_int = int(confinement_mode_key.replace('/', ''))

        # Convert to tensor
        confinement_mode_id_tensor = torch.tensor([confinement_mode_id_int], dtype=torch.int64)
        
        return signal_window, label, confinement_mode_id_tensor
    

def main(
        data_file: str|Path,
        confinement_data_file: str|Path,
        max_elms: int|Any = None,
        signal_window_size = 1024,
        experiment_name = 'experiment_default',
        # model
        initial_max_lr = 1e-3,
        layerwise_lr_decrement = 1.5,
        weight_decay = 1e-4,
        lr_scheduler_patience = 8,
        monitor_metric = None,
        do_dropout = False,
        dropout_percent = 0.05,
        use_optimizer = 'SGD',
        # loggers
        log_freq = 100,
        use_wandb = False,
        # callbacks
        early_stopping_min_delta = 1e-3,
        early_stopping_patience = 5,
        # trainer
        max_epochs = 2,
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        skip_train: bool = False,
        precision = None,
        # data
        batch_size = 64,
        fraction_validation = 0.12,
        fraction_test = 0.0,
        num_workers = 0,
        time_to_elm_quantile_min: float|Any = None,
        time_to_elm_quantile_max: float|Any = None,
        contrastive_learning: bool = True,
        min_pre_elm_time: float|Any = None,
        fir_hp_filter: float = 0.0,
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
        lr_scheduler_patience=lr_scheduler_patience,
        do_dropout=do_dropout,
        dropout_percent=dropout_percent,
        monitor_metric=monitor_metric,
        use_optimizer=use_optimizer,
    )
    monitor_metric = lit_model.monitor_metric
    ### callbacks
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

    if is_global_zero:
        print("Model Summary:")
        print(ModelSummary(lit_model, max_depth=-1))

    # exit()

    ### initialize trainer
    if precision is None:
        precision = '16-mixed' if torch.cuda.is_available() else 32
    trainer = Trainer(
        max_epochs = max_epochs,
        gradient_clip_val = gradient_clip_val,
        gradient_clip_algorithm = gradient_clip_algorithm,
        logger = loggers,
        log_every_n_steps = log_freq,
        callbacks = callbacks,
        enable_checkpointing = True,
        enable_progress_bar = False,
        enable_model_summary = False,
        precision = precision,
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=True,
        ) if world_size>1 else 'auto',
        num_nodes = num_nodes,
        use_distributed_sampler=False,
    )
    lit_model.save_hyperparameters({
        'gradient_clip_val': gradient_clip_val, 
        'gradient_clip_algorithm': gradient_clip_algorithm, 
        'precision': precision,
    })

    assert trainer.node_rank == node_rank
    assert trainer.world_size == world_size
    assert trainer.local_rank == local_rank
    assert trainer.global_rank == rank
    assert trainer.is_global_zero == is_global_zero

    ### data
    lit_datamodule = Data(
        signal_window_size=signal_window_size,
        data_file=data_file,
        confinement_data_file=confinement_data_file,
        max_elms=max_elms,
        batch_size=batch_size,
        fraction_test=fraction_test,
        fraction_validation=fraction_validation,
        num_workers=num_workers,
        time_to_elm_quantile_min=time_to_elm_quantile_min,
        time_to_elm_quantile_max=time_to_elm_quantile_max,
        contrastive_learning=contrastive_learning,
        is_global_zero=is_global_zero,
        min_pre_elm_time=min_pre_elm_time,
        fir_hp_filter=fir_hp_filter,
        max_shots_per_class=3,
        max_shots=12,
    )

    if skip_train is False:
        trainer.fit(lit_model, datamodule=lit_datamodule)
        if fraction_test:
            trainer.test(lit_model, lit_datamodule)

    if use_wandb:
        wandb.finish()

if __name__=='__main__':
    main(
        data_file='/global/homes/d/drsmith/scratch-ml/data/labeled_elm_events.hdf5',
        # data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        # data_file='/Users/drsmith/Documents/repos/bes-ml/bes_ml/small_elm_data.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=300,
        batch_size=128,
        max_epochs=2,
        fraction_validation=0.2,
        fraction_test=0.2,
        # num_workers=2,
        # time_to_elm_quantile_min=0.4,
        # time_to_elm_quantile_max=0.6,
        # contrastive_learning=True,
        # min_pre_elm_time=20,
        skip_train=False,
        # fir_hp_filter=5.0,
        # use_optimizer='sgd',
    )