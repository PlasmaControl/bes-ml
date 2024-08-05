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
import scipy.signal
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
    elm_classifier: bool = True
    conf_classifier: bool = False
    lr: float = 1e-3  # maximum LR used by first layer
    lr_scheduler_patience: int = 50
    lr_scheduler_threshold: float = 1e-3
    lr_warmup_epochs: int = 8
    # lr_layerwise_decrement: float = 1.
    weight_decay: float = 1e-6
    leaky_relu_slope: float = 2e-2
    monitor_metric: str = None
    use_optimizer: str = 'SGD'
    # feature_batchnorm: bool = True
    # task_batchnorm: bool = False

    def __post_init__(self):

        # init superclasses
        super().__init__()
        super(LightningModule, self).__post_init__()

        self.save_hyperparameters()
        if self.is_global_zero:
            print_fields(self)

        # single input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.make_feature_model()

        # task sub-models and metrics
        self.task_models = torch.nn.ModuleDict()
        self.task_metrics: dict[str, dict] = {}

        # Sub-model: ELM median time-to-ELM binary classifier
        if self.elm_classifier:
            task_name = 'elm_classifier'
            self.zprint(f"Task {task_name}")
            self.task_models[task_name] = self.make_mlp_classifier()
            self.task_metrics[task_name] = {
                'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                'f1_score': sklearn.metrics.f1_score,
                'precision_score': sklearn.metrics.precision_score,
                'recall_score': sklearn.metrics.recall_score,
                'mean_stat': torch.mean,
                'std_stat': torch.std,
            }
            if self.monitor_metric is None:
                self.monitor_metric = f'{task_name}/f1_score/val'

        # sub-model: Confinement mode multi-class classifier
        if self.conf_classifier:
            task_name = 'conf_classifier'
            self.zprint(f"Task {task_name}")
            self.task_models[task_name] = self.make_mlp_classifier(n_out=4)
            self.task_metrics[task_name] = {
                'ce_loss': torch.nn.functional.cross_entropy,
                'f1_score': sklearn.metrics.f1_score,
                'precision_score': sklearn.metrics.precision_score,
                'recall_score': sklearn.metrics.recall_score,
                'mean_stat': torch.mean,
                'std_stat': torch.std,
            }
            if self.monitor_metric is None:
                self.monitor_metric = f'{task_name}/f1_score/val'

        self.task_names = list(self.task_models.keys())

        self.zprint("Initializing model to uniform random weights and biases=0")
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                self.zprint(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith("weight"):
                if 'BatchNorm' in name:
                    self.zprint(f"  {name}: initialized to ones (numel {param.data.numel()})")
                    param.data.fill_(1)
                else:
                    n_in = np.prod(param.shape[1:])
                    sqrt_k = np.sqrt(3. / n_in)
                    self.zprint(f"  {name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")
                    param.data.uniform_(-sqrt_k, sqrt_k)
            else:
                raise ValueError

        if self.is_global_zero: 
            print("Batch evaluation (batch_size=128) with randn() data")
            example_batch_data = torch.randn(
                size=[128]+list(self.input_data_shape[1:]),
                dtype=torch.float32,
            )
            batch_input = {task: [example_batch_data] for task in self.task_names}
            batch_output = self(batch_input)
            for task, task_output in batch_output.items():
                print(f"  Task {task} output shape: {task_output.shape}")

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")
        return

    @staticmethod
    def param_count(model: LightningModule) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_feature_model(self) -> None:

        self.zprint("Feature space sub-model")

        feature_layer_dict = OrderedDict()

        conv_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 4, 'kernel': (1, 4, 4), 'stride': 1},
        )

        data_shape = self.input_data_shape
        self.zprint(f"  Input data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels: int|Any = None
        for i_layer, layer in enumerate(conv_layers):
            conv_layer_name = f"L{i_layer:02d}_Conv"
            conv = torch.nn.Conv3d(
                in_channels=1 if out_channels is None else out_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
            )
            n_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            self.zprint(f"  {conv_layer_name} kern {conv.kernel_size}  stride {conv.stride}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {np.prod(data_shape)})")
            out_channels = conv.out_channels
            feature_layer_dict[conv_layer_name] = conv
            feature_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)
            feature_layer_dict[f"L{i_layer:02d}_BatchNorm"] = torch.nn.BatchNorm3d(out_channels)

        feature_layer_dict['Flatten'] = torch.nn.Flatten()
        self.feature_model = torch.nn.Sequential(feature_layer_dict)
        self.feature_space_size = self.feature_model(torch.zeros(self.input_data_shape)).numel()

        self.zprint(f"  Feature sub-model parameters: {self.param_count(self.feature_model):,d}")
        self.zprint(f"  Feature space size: {self.feature_space_size}")

    def make_mlp_classifier(self, n_out: int = 1) -> torch.nn.Module:

        self.zprint("MLP classifier sub-model")

        mlp_layer_dict = OrderedDict()

        assert self.feature_space_size
        mlp_layer_sizes = (self.feature_space_size, 32, n_out)
        n_layers = len(mlp_layer_sizes)

        for i_layer in range(n_layers-1):
            mlp_layer_name = f"L{i_layer:02d}_FC"
            mlp_layer = torch.nn.Linear(
                in_features=mlp_layer_sizes[i_layer],
                out_features=mlp_layer_sizes[i_layer+1],
                bias=True if i_layer+1<n_layers-1 else False,
            )
            n_params = sum(p.numel() for p in mlp_layer.parameters() if p.requires_grad)
            self.zprint(f"  {mlp_layer_name}  in_features {mlp_layer.in_features}  out_features {mlp_layer.out_features}  parameters {n_params:,d}")
            mlp_layer_dict[mlp_layer_name] = mlp_layer
            if i_layer+1 < n_layers-1:
                mlp_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)

        mlp_classifier = torch.nn.Sequential(mlp_layer_dict)

        self.zprint(f"  MLP sub-model parameters: {self.param_count(mlp_classifier):,d}")

        return mlp_classifier

    def configure_optimizers(self):
        self.zprint(f"Using {self.use_optimizer.upper()} optimizer")
        optim_kwargs = {
            'params': self.parameters(),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        if self.use_optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(momentum=0.2, **optim_kwargs)
        elif self.use_optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(**optim_kwargs)
        else:
            raise ValueError

        lr_reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            min_lr=1e-4,
            verbose=True,
        )
        lr_warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.1,
            total_iters=self.lr_warmup_epochs,
            verbose=True,
        )
        return_optim_list = [self.optimizer]
        return_lr_scheduler_list = [
            {'scheduler': lr_reduce_on_plateau, 'monitor': self.monitor_metric},
            lr_warm_up, 
        ]
        return (return_optim_list, return_lr_scheduler_list)

    def training_step(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor:
        return self.update_step(
            batch, 
            batch_idx, 
            stage='train',
            dataloader_idx=dataloader_idx,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        self.update_step(
            batch, 
            batch_idx, 
            stage='val',
            dataloader_idx=dataloader_idx,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        self.update_step(
            batch, 
            batch_idx, 
            stage='test',
            dataloader_idx=None if isinstance(batch, dict) else dataloader_idx,
        )

    def update_step(
            self, 
            batch: dict|list, 
            batch_idx = None, 
            dataloader_idx = None,
            stage: str = '', 
    ) -> torch.Tensor:
        sum_loss = torch.Tensor([0.0])
        model_outputs = self(batch)
        for task in model_outputs:
            task_outputs = model_outputs[task]
            metrics = self.task_metrics[task]
            if task == 'elm_classifier' and dataloader_idx in [None, 0]:
                labels = batch[task][1][0.5] if isinstance(batch, dict) else batch[1][0.5]
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs.reshape_as(labels),
                            target=labels.type_as(task_outputs),
                        )
                        sum_loss = sum_loss + metric_value if sum_loss else metric_value
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            y_pred=(task_outputs.detach().cpu() >= 0.0).type(torch.int), 
                            y_true=labels.detach().cpu(),
                            zero_division=0,
                        )
                        if self.current_epoch<10:
                            metric_value /= 10
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs)
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)
            elif task == 'conf_classifier' and dataloader_idx in [None, 1]:
                labels = batch[task][1] if isinstance(batch, dict) else batch[1]
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs,
                            target=labels.flatten(),
                        )
                        sum_loss = sum_loss + metric_value if sum_loss else metric_value
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            y_pred=(task_outputs > 0.0).type(torch.int).detach().cpu(), 
                            y_true=torch.nn.functional.one_hot(
                                labels.flatten().detach().cpu(),
                                num_classes=4,
                            ),
                            zero_division=0,
                            average='macro',
                        )
                        if self.current_epoch<10:
                            metric_value /= 10
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs)
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)
        return sum_loss

    def forward(
            self, 
            batch: dict|list, 
    ) -> dict[str,torch.Tensor]:
        results = {}
        for task in self.task_models:
            if isinstance(batch, dict):
                # dict batches for training
                results[task] = self.task_models[task](self.feature_model(batch[task][0]))
            else:
                # list batches for val/test/predict
                results[task] = self.task_models[task](self.feature_model(batch[0]))
        return results

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with global step {self.trainer.global_step} ****")

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        self.zprint(f"Fit time: {delt/60:0.1f} min")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.t_train_epoch_start
        global_time = time.time() - self.t_fit_start
        epoch_steps = self.global_step-self.s_train_epoch_start
        if self.is_global_zero and self.global_step > 0:
            line =  f"Ep {self.current_epoch:03d}  "
            line += f"ep/gl steps {epoch_steps:,d}/{self.global_step:,d}  "
            line += f"ep/gl minutes: {epoch_time/60:.2f}/{global_time/60:.2f}  " 
            for task in self.task_models:
                train_score = self.trainer.logged_metrics[f'{task}/f1_score/train']
                val_score = self.trainer.logged_metrics[f'{task}/f1_score/val']
                line += f"{task} tr/val score {train_score:.4f}/{val_score:.4f}  "
            print(line)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def setup(self, stage=None):  # fit, validate, test, or predict
        assert self.is_global_zero == self.trainer.is_global_zero
        if self.is_global_zero:
            assert self.global_rank == 0

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.trainer.world_size > 1:
            print(f"Rank {self.trainer.global_rank}: {text}")
        else:
            print(text)

@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    elm_data_file: str|Path|Any = None
    confinement_data_file: str|Path|Any = None
    elm_classifier: bool = True
    conf_classifier: bool = False
    max_elms: int|Any = None
    batch_size: int = 256
    stride_factor: int = 8
    num_workers: int|Any = None
    outlier_value: float = 6
    normalized_signal_outlier_value: float = 8
    fraction_validation: float = 0.12
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = None  # seed for ELM index shuffling; must be same across processes
    time_to_elm_quantile_min: float|Any = None
    time_to_elm_quantile_max: float|Any = None
    contrastive_learning: bool = False
    min_pre_elm_time: float|Any = None
    epochs_per_batch_size_reduction: int = None
    max_pow2_batch_size_reduction: int = 2
    fir_taps: int = 501  # Number of taps in the filter
    fir_bp_low: float|Any = None  # bandpass filter cut-on freq in kHz
    fir_bp_high: float|Any = None  # bandpass filter cut-off freq in kHz
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
    test_only: bool = False
    n_rows: int = 8
    n_cols: int = 8
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    max_confinement_event_length: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()

        self.trainer: Trainer|Any = None
        self.batch_size_per_rank: int = 0
        self.a_coeffs = self.b_coeffs = None

        if self.num_workers is None:
            self.num_workers = 8 if self.trainer.world_size>1 else 0

        self.tasks = []

        if self.elm_classifier:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists()
            self.tasks.append('elm_classifier')

            self.global_shot_split: dict[str,np.ndarray] = {}
            self.global_elm_split: dict[str,Sequence] = {}
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.elm_raw_signal_mean: float|Any = None
            self.elm_raw_signal_stdev: float|Any = None

        if self.conf_classifier:
            self.confinement_data_file = Path(self.confinement_data_file).absolute()
            assert self.confinement_data_file.exists()
            self.tasks.append('conf_classifier')

            self.confinement_datasets: dict[str,torch.utils.data.Dataset] = {}
            self.global_confinement_shot_split: dict[str,Sequence] = {}
            self.stage_to_events: dict = {}
            self.confinement_raw_signal_mean: float|Any = None
            self.confinement_raw_signal_stdev: float|Any = None
            self.confinement_train_dataloader: torch.utils.data.DataLoader|Any = None
            self.confinement_mask_lb: float|Any = None
            self.confinement_mask_ub: float|Any = None
            self.train_confinement_events: list = []
            self.validation_confinement_events: list = []
            self.test_confinement_events: list = []

        if self.is_global_zero:
            print_fields(self)

        self.state_items = []
        if self.elm_classifier:
            self.state_items.extend([
                'global_elm_split',
                'global_shot_split',
                'elm_raw_signal_mean',
                'elm_raw_signal_stdev',
                'time_to_elm_quantiles',
            ])
        if self.conf_classifier:
            self.state_items.extend([
                'global_confinement_shot_split',
                'confinement_raw_signal_mean',
                'confinement_raw_signal_stdev',
                'confinement_mask_lb',
                'confinement_mask_ub',
            ])

        for item in self.state_items:
            assert hasattr(self, item)

    def barrier(self) -> None:
        if self.trainer.world_size > 1:
            self.trainer.strategy.barrier()

    def broadcast(self, obj):
        if self.trainer.world_size == 0:
            return obj
        else:
            obj = self.trainer.strategy.broadcast(obj)
            return obj

    def prepare_data(self):
        self.zprint(f"**** Prepare data")
        if self.seed is None:
            self.seed = np.random.default_rng().integers(0, 2**32-1)
        self.rng = np.random.default_rng(self.seed)
        # make global shot split and global ELM split
        self.zprint("  Creating global ELM data split")
        with h5py.File(self.elm_data_file, 'r') as root:
            datafile_shots = set([int(shot_key) for shot_key in root['shots']])
            self.zprint(f"    Shots in data file: {len(datafile_shots):,d}")
            datafile_shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(datafile_shots ^ datafile_shots_from_elms) == 0
            datafile_shots = list(datafile_shots)
            datafile_elms = [int(elm_key) for elm_key in root['elms']]
            self.zprint(f"    ELMs in data file: {len(datafile_elms):,d}")
            # limit max ELMs
            if self.max_elms and len(datafile_elms) > self.max_elms:
                self.rng.shuffle(datafile_elms)
                datafile_elms = datafile_elms[:self.max_elms]
                datafile_shots = [int(root['elms'][f"{elm_index:06d}"].attrs['shot']) for elm_index in datafile_elms]
                datafile_shots = list(set(datafile_shots))
                self.zprint(f"    ELMs/shots for analysis: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # shuffle shots in dataset
            self.zprint(f"    Shuffling global shots")
            self.rng.shuffle(datafile_shots)
            # global shot split
            self.zprint("    Global shot split")
            n_test_shots = int(self.fraction_test * len(datafile_shots))
            n_validation_shots = int(self.fraction_validation * len(datafile_shots))
            self.global_shot_split['test'], self.global_shot_split['validation'], self.global_shot_split['train'] = \
                np.split(datafile_shots, [n_test_shots, n_test_shots+n_validation_shots])
            for stage in ['train','validation','test']:
                self.zprint(f"      {stage.upper()}: Global shots: {self.global_shot_split[stage].size} ({self.global_shot_split[stage].size/len(datafile_shots)*1e2:.1f}%)")
            # global ELM split
            for stage in ['train','validation','test']:
                self.global_elm_split[stage] = [
                    i_elm for i_elm in datafile_elms
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] in self.global_shot_split[stage]
                ]
                self.zprint(f"      {stage.upper()}: Global ELM count {len(self.global_elm_split[stage]):,d} ({len(self.global_elm_split[stage])/len(datafile_elms)*1e2:.1f}%)")

    def setup(self, stage: str):
        t_tmp = time.time()
        self.zprint(f"**** Setup stage: {stage.upper()}")

        assert stage in ['fit', 'test', 'predict']
        assert self.is_global_zero == self.trainer.is_global_zero

        assert self.batch_size % self.trainer.world_size == 0
        self.batch_size_per_rank = self.batch_size // self.trainer.world_size
        self.zprint(f"  Global batch size: {self.batch_size}")
        self.zprint(f"  Rank batch size: {self.batch_size_per_rank}")

        if self.fir_bp_low is None and self.fir_bp_high is None:
            self.zprint("  Using raw BES signals with no FIR filter")
        else:
            self.zprint(f"  FIRfilter with f_low-f_high: {self.fir_bp_low} - {self.fir_bp_high} kHz")
            if self.fir_bp_low and self.fir_bp_high:
                pass_zero = 'bandpass'
                cutoff = [self.fir_bp_low, self.fir_bp_high]
            elif self.fir_bp_low:
                pass_zero = 'highpass'
                cutoff = self.fir_bp_low
            elif self.fir_bp_high:
                pass_zero = 'lowpass'
                cutoff = self.fir_bp_high
            self.b_coeffs = scipy.signal.firwin(
                numtaps=self.fir_taps,  # must be odd
                cutoff=cutoff,  # transition width in kHz
                pass_zero=pass_zero,
                fs=1e3,  # f_sample in kHz
            )
            self.a_coeffs = np.zeros_like(self.b_coeffs)
            self.a_coeffs[0] = 1

        stages = ['train', 'validation'] if stage == 'fit' else [stage]
        self.zprint(f"  Data setup for stages: {stages}")

        if self.elm_classifier:
            t_tmp = time.time()
            self.zprint("**** ELM data setup")
            self.global_elm_split = self.broadcast(self.global_elm_split)
            self.global_shot_split = self.broadcast(self.global_shot_split)
            for st in stages:
                self._prepare_elm_data_for_stage(st)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
            self.barrier()


        if self.conf_classifier:
            self.zprint("**** Confinement data preparation")
            if 'train' not in self.global_confinement_shot_split:
                self._get_confinement_events_and_split()
            else:
                self.zprint("  Reusing saved global confinement data split")
            for st in stages:
                self._setup_confinement_data(st)
            self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")

    def _prepare_elm_data_for_stage(self, st: str):
        self.zprint(f"  ELM {st.upper()} data setup")
        if st in self.elm_datasets and isinstance(self.elm_datasets[st], torch.utils.data.Dataset):
            self.zprint(f"    Using saved dataset")
            return
        elm_indices = self.global_elm_split[st]
        n_elms = len(elm_indices)
        self.zprint(f"    ELM count: {n_elms}")
        assert n_elms > 0
        stage_sw_metadata: list = []
        outliers = 0
        skipped_short_pre_elm_time = 0
        if self.outlier_value:
            self.zprint(f"    Removing outliers with max(abs(signal windows)) > {self.outlier_value:.3f} V")
        with h5py.File(self.elm_data_file, 'r') as h5_file:
            elms: h5py.Group = h5_file['elms']
            for i_elm, elm_index in enumerate(elm_indices):
                if i_elm%(n_elms//10) == 0:
                    self.zprint(f"    Reading ELM event {i_elm:04d}/{len(elm_indices):04d}")
                elm_event: h5py.Group = elms[f"{elm_index:06d}"]
                shot = int(elm_event.attrs['shot'])
                assert elm_event["bes_signals"].shape[0] == 64
                assert elm_event['bes_time'].size == elm_event["bes_signals"].shape[1]
                bes_time = np.array(elm_event['bes_time'], dtype=np.float32)
                t_start: float = elm_event.attrs['t_start']
                t_stop: float = elm_event.attrs['t_stop'] - 0.05
                if self.min_pre_elm_time and (t_stop-t_start) < self.min_pre_elm_time:
                    skipped_short_pre_elm_time += 1
                    continue
                i_start: int = np.flatnonzero(bes_time >= t_start)[0]
                i_stop: int = np.flatnonzero(bes_time <= t_stop)[-1]
                i_window_stop = i_stop
                signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (bes_time, pol, rad)
                assert signals.shape[0] == bes_time.size
                assert (signals.shape[1] == 8) and (signals.shape[2] == 8)
                while True:
                    i_window_start = i_window_stop - self.signal_window_size
                    if i_window_start < i_start:
                        break  # break while loop
                    if self.outlier_value:  # raw signal outlier filter
                        signal_window = signals[i_window_start:i_window_stop, ...]
                        assert signal_window.shape[0] == self.signal_window_size
                        if np.abs(signal_window).max() > self.outlier_value:
                            i_window_stop -= self.signal_window_size // self.stride_factor
                            outliers += 1
                            continue
                    stage_sw_metadata.append({
                        'elm_index': elm_index,
                        'shot': shot,
                        'i_t0': i_window_start,
                        'time_to_elm': bes_time[i_stop] - bes_time[i_window_stop]
                    })
                    i_window_stop -= self.signal_window_size // self.stride_factor

        self.zprint(f"    Skipped ELMs for short pre-ELM time: {skipped_short_pre_elm_time}")

        n_signal_windows = len(stage_sw_metadata)
        self.zprint(f"    Signal windows (unprocessed): {n_signal_windows:,d}  ({outliers:,d} outliers removed)")

        # stats
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_interval = np.max([self.stride_factor, len(stage_sw_metadata)//int(10e3)])
        last_elm_index = -1
        with h5py.File(self.elm_data_file) as root:
            for sw in stage_sw_metadata[::stat_interval]:
                elm_index = sw['elm_index']
                if elm_index != last_elm_index:
                    elm_event: h5py.Group = root['elms'][f'{elm_index:06d}']
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                last_elm_index = elm_index
                i_t0 = sw['i_t0']
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
        self.zprint(f"    Stage raw signal stats:  mean {mean:.3f}  stdev {stdev:.3f}  exkurt {exkurt:.3f}  min/max {signal_min:.3f}/{signal_max:.3f}")
        self.barrier()
        if self.is_global_zero and st == 'train' and not self.elm_raw_signal_mean:
            self.zprint(f"    Using {st.upper()} for standardizing mean and stdev")
            self.elm_raw_signal_mean = mean.item()
            self.elm_raw_signal_stdev = stdev.item()
            self.save_hyperparameters({
                'raw_signal_mean': self.elm_raw_signal_mean,
                'raw_signal_stdev': self.elm_raw_signal_stdev,
            })
        self.barrier()
        self.elm_raw_signal_mean = self.broadcast(self.elm_raw_signal_mean)
        self.elm_raw_signal_stdev = self.broadcast(self.elm_raw_signal_stdev)
        self.rprint(f"    Standarizing signals with mean {self.elm_raw_signal_mean:.3f} and std {self.elm_raw_signal_stdev:.3f}")

        # time-to-ELM quantiles
        if st == 'train' and not self.time_to_elm_quantiles:
            quantiles = [0.5]
            time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in stage_sw_metadata]
            quantile_values = np.quantile(time_to_elm_labels, quantiles)
            self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
            self.save_hyperparameters({
                'time_to_elm_quantiles': self.time_to_elm_quantiles,
            })
            self.zprint(f"    Time-to-ELM quantiles for binary labels:")
            for q, qval in self.time_to_elm_quantiles.items():
                self.zprint(f"      Quantile {q:.2f}: {qval:.1f} ms")
            self.barrier()
            rank0_values = self.broadcast(quantile_values)
            assert np.array_equal(rank0_values, quantile_values)
        assert self.time_to_elm_quantiles

        # restrict data according to quantiles
        if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
            time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in stage_sw_metadata])
            time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
            if self.contrastive_learning:
                self.zprint(f"    Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                for i in np.arange(len(stage_sw_metadata)-1, -1, -1, dtype=int):
                    if (stage_sw_metadata[i]['time_to_elm'] > time_to_elm_min) and \
                        (stage_sw_metadata[i]['time_to_elm'] < time_to_elm_max):
                        stage_sw_metadata.pop(i)
            else:
                self.zprint(f"    Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                for i in np.arange(len(stage_sw_metadata)-1, -1, -1, dtype=int):
                    if (stage_sw_metadata[i]['time_to_elm'] < time_to_elm_min) or \
                        (stage_sw_metadata[i]['time_to_elm'] > time_to_elm_max):
                        stage_sw_metadata.pop(i)
        remainder = len(stage_sw_metadata) % self.trainer.world_size
        if remainder:
            stage_sw_metadata = stage_sw_metadata[:-remainder]
        assert len(stage_sw_metadata) % self.trainer.world_size == 0
        n_signal_windows = len(stage_sw_metadata)
        self.zprint(f"    Stage signal windows (final): {n_signal_windows:,d}")
        self.zprint(f"    Batches per epoch: {n_signal_windows/self.batch_size:,.1f}")

        # split signal windows across ranks
        rankwise_sw_split = np.array_split(stage_sw_metadata, self.trainer.world_size)
        for ir in range(self.trainer.world_size):
            rank0_values = self.broadcast(rankwise_sw_split[ir][:5])
            for i, d in enumerate(rank0_values):
                assert d['elm_index'] == rankwise_sw_split[ir][i]['elm_index']
        self.zprint("    Consistent signal window split across ranks")
        sw_for_rank = list(rankwise_sw_split[self.trainer.global_rank])
        self.rprint(f"    Signal windows {len(sw_for_rank):,d}")

        # get rank-wise ELM signals
        elms_for_rank = np.unique(np.array([item['elm_index'] for item in sw_for_rank],dtype=int))
        elm_to_signals = {}
        with h5py.File(self.elm_data_file) as root:
            for elm_index in elms_for_rank:
                elm_group: h5py.Group = root['elms'][f"{elm_index:06d}"]
                signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                if self.b_coeffs is not None:
                    signals = np.array(
                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                        dtype=np.float32,
                    )
                signals = np.transpose(signals).reshape(1, -1, 8, 8)  # reshape to (time, pol, rad)
                elm_to_signals[elm_index] = (signals - self.elm_raw_signal_mean) / self.elm_raw_signal_stdev
        assert len(elm_to_signals) == len(elms_for_rank)

        # rank-wise datasets
        if st in ['train', 'validation', 'test']:
            self.elm_datasets[st] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                signal_list=elm_to_signals,
                quantile_min=self.time_to_elm_quantile_min,
                quantile_max=self.time_to_elm_quantile_max,
                contrastive_learning=self.contrastive_learning,
            )
        
        if st in ['test', 'predict']:
            pass

    def _get_confinement_events_and_split(self):
        self.zprint("  Creating global confinement data split")
        if self.bad_shots is None:
            self.bad_shots = []  # Initialize to empty list if None
        check_bounds = lambda value, bounds: bounds[0] <= value <= bounds[1] if bounds else True
        global_shot_to_events: dict[int,list] = {}
        global_class_to_shots: list[list] = [[] for _ in range(self.num_classes)]
        global_class_duration: list[int] = [0] * self.num_classes
        global_class_to_events: list[int] = [0] * self.num_classes
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        missing_inboard = bad_inboard = 0
        with h5py.File(self.confinement_data_file) as root:
            for shot in root:
                # shot exclusions
                if shot in self.bad_shots:
                    continue
                inboard_order = root[shot].attrs.get("inboard_column_channel_order", None)
                if inboard_order is None or len(inboard_order)==0:
                    missing_inboard += 1
                    continue
                if not np.array_equal(inboard_order, np.arange(8, dtype=int)*8+1):
                    bad_inboard += 1
                    continue
                metadata = {
                    'r_avg': root[shot].attrs.get('r_avg'),
                    'z_avg': root[shot].attrs.get('z_avg'),
                    'delz_avg': root[shot].attrs.get('delz_avg')
                }
                if not all(
                    check_bounds(metadata[key], self.metadata_bounds[key]) 
                    for key in ['r_avg', 'z_avg', 'delz_avg'] 
                    if key in self.metadata_bounds
                ):
                    if metadata['r_avg'] is None or not check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
                        r_avg_exclusions += 1
                    if metadata['z_avg'] is None or not check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
                        z_avg_exclusions += 1
                    if metadata['delz_avg'] is None or not check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
                        delz_avg_exclusions += 1
                    continue
                # loop over events in shot
                events: list[dict] = []
                for event_key in root[shot]:
                    event = root[shot][event_key]
                    if 'labels' not in event:
                        continue
                    event_label: int = event['labels'][0].item()
                    assert event_label < self.num_classes
                    event_duration: int = event['signals'].shape[1]
                    if self.max_confinement_event_length and event_duration>self.max_confinement_event_length:
                        event_duration = self.max_confinement_event_length
                    if event_duration < self.signal_window_size:
                        continue
                    events.append({
                        'shot': int(shot),
                        'event': int(event_key),
                        'label': event_label,
                        'duration': event_duration,
                    })
                    global_class_to_shots[event_label].append(int(shot))
                    global_class_duration[event_label] += event_duration
                    global_class_to_events[event_label] += 1
                if not events:
                    continue
                global_shot_to_events[int(shot)] = events
                # {
                #     'events': shot_events, 
                #     'metadata': metadata,
                # }
        global_class_to_shots = [list(set(item)) for item in global_class_to_shots]
        # BES location exclusions
        self.zprint(f"    missing inboard shot exclusions: {missing_inboard}")
        self.zprint(f"    bad inboard shot exclusions: {bad_inboard}")
        self.zprint(f"    r_avg shot exclusions: {r_avg_exclusions}")
        self.zprint(f"    z_avg shot exclusions: {z_avg_exclusions}")
        self.zprint(f"    delz_avg shot exclusions: {delz_avg_exclusions}")
        # data read
        self.zprint("  Data file summary (some shots maybe excluded)")
        self.zprint(f"    Shots: {len(global_shot_to_events)}")
        for i in range(self.num_classes):
            self.zprint(f"      Class {i}:  shots {len(global_class_to_shots[i])}  events {global_class_to_events[i]}  duration {global_class_duration[i]:,d}")
            assert global_class_duration[i]
        # Forced shots
        forced_test_shots_data = {}
        forced_validation_shots_data = {}
        if self.force_test_shots:
            for shot_number in self.force_test_shots:
                forced_test_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        if self.force_validation_shots:
            for shot_number in self.force_validation_shots:
                forced_validation_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        if self.max_shots_per_class is not None:
            for i_class in range(len(global_class_to_shots)-1, -1, -1):
                class_shots = global_class_to_shots[i_class]
                if len(class_shots) > self.max_shots_per_class:
                    # down-select shots for each class
                    global_class_to_shots[i_class] = self.rng.choice(
                        a=class_shots, 
                        size=self.max_shots_per_class, 
                        replace=False,
                    ).tolist()
            global_shot_to_events = {shot: global_shot_to_events[shot] for shots in global_class_to_shots for shot in shots}
        shot_numbers = list(global_shot_to_events.keys())
        if self.test_only:
            return

        # split global data into train, val, test
        good_split = False
        while good_split == False:
            self.rng.shuffle(shot_numbers)
            good_split = True
            self.global_confinement_shot_split = {st:[] for st in ['train','validation','test']}
            self.global_confinement_shot_split['train'], _test_val_shots = train_test_split(
                shot_numbers, 
                test_size=self.fraction_test + self.fraction_validation, 
                random_state=self.rng.integers(0, 2**32-1),
            )
            if self.fraction_test:
                self.global_confinement_shot_split['test'], self.global_confinement_shot_split['validation'] = train_test_split(
                    _test_val_shots,
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    random_state=self.rng.integers(0, 2**32-1),
                )
            else:
                self.global_confinement_shot_split['validation'] = _test_val_shots
                self.global_confinement_shot_split['test'] = []

            if forced_test_shots_data or forced_validation_shots_data:
                global_shot_to_events.update(forced_validation_shots_data)
                global_shot_to_events.update(forced_test_shots_data)
                self.global_confinement_shot_split['validation'].extend(list(forced_validation_shots_data.keys()))
                self.global_confinement_shot_split['test'].extend(list(forced_test_shots_data.keys()))

            # Assign events to datasets
            self.zprint("  Final data for computation")
            self.stage_to_events = {}
            for st in self.global_confinement_shot_split:
                self.stage_to_events[st] = [event for shot in self.global_confinement_shot_split[st] for event in global_shot_to_events[shot]['events']]
                if len(self.global_confinement_shot_split[st]) == 0:
                    self.zprint(f"    {st.capitalize()} data: {len(self.global_confinement_shot_split[st])} shots and {len(self.stage_to_events[st])} events")
                    continue
                class_to_shots = [[] for _ in range(self.num_classes)]
                class_to_events = [0] * self.num_classes
                class_to_duration = [0] * self.num_classes
                for event in self.stage_to_events[st]:
                    class_to_shots[event['label']].append(event['shot'])
                    class_to_events[event['label']] += 1
                    class_to_duration[event['label']] += event['duration']
                class_to_shots = [list(set(l)) for l in class_to_shots]
                if (0 in class_to_events) or good_split == False:
                    good_split = False
                    self.zprint("        Bad split, re-running")
                    continue
                self.zprint(f"    {st.capitalize()} data: {len(self.global_confinement_shot_split[st])} shots and {len(self.stage_to_events[st])} events")
                for i in range(self.num_classes):
                    self.zprint(f"      Class {i}: {len(class_to_shots[i])} shots, {class_to_events[i]} events, {class_to_duration[i]:,d} timepoints")
                    assert len(class_to_shots[i]) > 0
            # set predict dataset
            self.global_confinement_shot_split['predict'] = self.global_confinement_shot_split['test']
            self.stage_to_events['predict'] = self.stage_to_events['test']
        return

    def _setup_confinement_data(self, stage: str):
        t_tmp = time.time()
        self.zprint(f"  Stage {stage.upper()}: Setup confinement data")
        # if stage == 'train' and self.trainer.world_size > 1:
        #     self.zprint("    Creating chunks")
        #     chunked_events = [[] for _ in range(self.trainer.world_size)]
        #     chunk_durations = [0] * self.trainer.world_size
        #     # Iterate over the indices, sorted by time from largest to smallest
        #     sorted_indices = sorted(range(len(events)), key=lambda i: events[i]['duration'], reverse=True)
        #     for i_ev in sorted_indices:
        #         # Find the chunk with the shortest total time so far
        #         min_time_chunk_idx = min(range(self.trainer.world_size), key=lambda i: chunk_durations[i])
        #         # Add this index to that chunk
        #         chunked_events[min_time_chunk_idx].append(events[i_ev])
        #         # Update the total time for that chunk
        #         chunk_durations[min_time_chunk_idx] += events[i_ev]['duration']
        #     for i, (chunk, chunk_time) in enumerate(zip(chunked_events, chunk_durations)):
        #         self.zprint(f"    Chunk {i} events: {len(chunk)}, total time: {chunk_time:.1f}")
        #     # Determine the chunk for this GPU
        #     chunk_events = chunked_events[self.trainer.global_rank]
        # else:
        #     chunk_events = events
        # chunk_events = events

        # dataset = self._load_and_preprocess_confinement_data(events, stage)
        events = self.stage_to_events[stage]
        event_data = []
        # time_count = sum([event['duration'] for event in events])
        # packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
        # start_index = 0

        with h5py.File(self.confinement_data_file, 'r') as root:
            for i, event_md in enumerate(events):
                shot = event_md['shot']
                event = event_md['event']
                if len(events) >= 10 and i % (len(events)//10) == 0:
                    self.zprint(f"    Reading event {i:04d}/{len(events):04d}")
                event_group = root[str(shot)][str(event)]
                # Retrieve signals and reshape according to inboard_order
                labels = np.array(event_group["labels"], dtype=int)
                # signals = np.array(event_group["signals"][:, :], dtype=np.float32)
                if self.max_confinement_event_length and labels.size>self.max_confinement_event_length:
                    labels = labels[:self.max_confinement_event_length]
                    # signals = signals[:,:self.max_confinement_event_length]
                # signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                # if self.b_coeffs is not None and signals.shape[0] > self.fir_taps:
                #     signals = np.array(
                #         scipy.signal.lfilter(
                #             x=signals,
                #             a=self.a_coeffs,
                #             b=self.b_coeffs,
                #         ),
                #         dtype=np.float32,
                #     )
                # packaged_signals[start_index:start_index + signals.shape[0]] = signals
                # start_index += signals.shape[0]
                valid_t0 = np.zeros(labels.size, dtype=int)
                first_valid_signal_window_start_index = self.signal_window_size - 1
                valid_t0[first_valid_signal_window_start_index::self.signal_window_size//8] = 1
                event_data.append({
                    'labels': labels, 
                    'valid_t0': valid_t0,
                    'confinement_mode_key': f"{shot}/{event}",
                    'shot': shot,
                    'time': event,
                })

        self.zprint(f"    Time for confinement data read: {time.time()-t_tmp:.1f} s")

        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in event_data], axis=0)
        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in event_data], axis=0)
        assert packaged_labels.size == packaged_valid_t0.size
        packaged_window_start = []
        index = 0
        for confinement_mode in event_data:
            packaged_window_start.append(index)
            index += confinement_mode['labels'].size
        packaged_window_start = np.array(packaged_window_start, dtype=int)
        assert len(event_data) == len(packaged_window_start)

        packaged_confinement_mode_key = np.array(
            [confinement_mode['confinement_mode_key'] for confinement_mode in event_data],
            dtype=str,
        )
        assert len(event_data) == len(packaged_confinement_mode_key)

        # valid t0 indices
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        self.zprint(f"  Stage {stage.upper()}: Raw data stats")
        # assert packaged_labels.size == packaged_signals.shape[0]

        del event_data

        def _get_statistics2(sample_indices: np.ndarray, signals: np.ndarray) -> dict:
            signal_min = np.array(np.inf)
            signal_max = np.array(-np.inf)
            n_bins = 200
            cummulative_hist = np.zeros(n_bins, dtype=int)
            stat_samples = int(10e3)
            stat_interval = np.max([1, sample_indices.size//stat_samples])
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
            self.zprint(f"    Stats: mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} min/max {signal_min:.3f}/{signal_max:.3f}")
            return {
                'count': sample_indices.size,
                'min': signal_min,
                'max': signal_max,
                'mean': mean,
                'stdev': stdev,
                'exkurt': exkurt,
            }

        # standardize signals based on training data
        stats = _get_statistics2(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )
        if self.is_global_zero and stage == 'train' and not self.confinement_raw_signal_mean:
            self.zprint(f"    Using {stage.upper()} for standarizing mean and stdev")
            self.confinement_raw_signal_mean = stats['mean']
            self.confinement_raw_signal_stdev = stats['stdev']
            self.save_hyperparameters({
                'signal_mean': self.confinement_raw_signal_mean.item(),
                'signal_stdev': self.confinement_raw_signal_stdev.item(),
            })
        self.barrier()
        self.confinement_raw_signal_mean = self.broadcast(self.confinement_raw_signal_mean)
        self.confinement_raw_signal_stdev = self.broadcast(self.confinement_raw_signal_stdev)
        self.rprint(f"  Stage {stage.upper()}: Standarizing signals with mean {self.confinement_raw_signal_mean:.3f} and std {self.confinement_raw_signal_stdev:.3f}")
        packaged_signals = (packaged_signals - self.confinement_raw_signal_mean) / self.confinement_raw_signal_stdev

        self.rprint(f"  Stage {stage.upper()}: Valid t0 indices: {len(packaged_valid_t0_indices):,d}")
        self.zprint(f"  Stage {stage.upper()}: Batches per epoch: {len(packaged_valid_t0_indices)/self.batch_size:.1f}")
        self.zprint(f"  Stage {stage.upper()}: Load/preprocess confinement data time: {time.time()-t_tmp:.1f} s")

        if stage == 'train':
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
        elif stage in ['validation', 'test']:
            self.confinement_datasets[stage] = Confinement_TrainValTest_Dataset(
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
        elif stage == 'predict':
            del self.confinement_train_dataloader
            del self.confinement_datasets['validation']
        else:
            raise ValueError
            
        gc.collect()
        torch.cuda.empty_cache()
        self.rprint(f'The CPU usage is: {psutil.cpu_percent(4)}')
        self.rprint(f'RAM memory % used: {psutil.virtual_memory()[2]}')
        self.rprint(f'RAM Used (GB): {psutil.virtual_memory()[3]/1000000000}')    
        # Store the DataLoader for this GPU
        if stage == 'train':
            self.confinement_train_dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=True,             
                num_workers=self.num_workers,
                persistent_workers=(self.num_workers > 0),
                drop_last=True,
            )

    # def _load_and_preprocess_confinement_data(self, events, stage):
        # t_tmp = time.time()
        # self.zprint(f"  Stage {stage.upper()}: load and pre-process data")
        # event_data = []
        # # time_count = sum([event['duration'] for event in events])
        # # packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
        # # start_index = 0

        # with h5py.File(self.confinement_data_file, 'r') as root:
        #     for i, event in enumerate(events):
        #         shot = event['shot']
        #         event = event['event']
        #         if len(events) >= 10 and i % (len(events)//10) == 0:
        #             self.zprint(f"    Reading event {i:04d}/{len(events):04d}")
        #         event_group = root[str(shot)][str(event)]
        #         # Retrieve signals and reshape according to inboard_order
        #         labels = np.array(event_group["labels"], dtype=int)
        #         # signals = np.array(event_group["signals"][:, :], dtype=np.float32)
        #         if self.max_confinement_event_length and labels.size>self.max_confinement_event_length:
        #             labels = labels[:self.max_confinement_event_length]
        #             # signals = signals[:,:self.max_confinement_event_length]
        #         # signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
        #         # if self.b_coeffs is not None and signals.shape[0] > self.fir_taps:
        #         #     signals = np.array(
        #         #         scipy.signal.lfilter(
        #         #             x=signals,
        #         #             a=self.a_coeffs,
        #         #             b=self.b_coeffs,
        #         #         ),
        #         #         dtype=np.float32,
        #         #     )
        #         # packaged_signals[start_index:start_index + signals.shape[0]] = signals
        #         # start_index += signals.shape[0]
        #         valid_t0 = np.zeros(labels.size, dtype=int)
        #         first_valid_signal_window_start_index = self.signal_window_size - 1
        #         valid_t0[first_valid_signal_window_start_index::self.signal_window_size//8] = 1
        #         event_data.append({
        #             'labels': labels, 
        #             'valid_t0': valid_t0,
        #             'confinement_mode_key': f"{shot}/{event}",
        #             'shot': shot,
        #             'time': event,
        #         })

        # self.zprint(f"    Time for confinement data read: {time.time()-t_tmp:.1f} s")

        # packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in event_data], axis=0)
        # packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in event_data], axis=0)
        # assert packaged_labels.size == packaged_valid_t0.size
        # packaged_window_start = []
        # index = 0
        # for confinement_mode in event_data:
        #     packaged_window_start.append(index)
        #     index += confinement_mode['labels'].size
        # packaged_window_start = np.array(packaged_window_start, dtype=int)
        # assert len(event_data) == len(packaged_window_start)

        # packaged_confinement_mode_key = np.array(
        #     [confinement_mode['confinement_mode_key'] for confinement_mode in event_data],
        #     dtype=str,
        # )
        # assert len(event_data) == len(packaged_confinement_mode_key)
        # del event_data

        # def _get_statistics2(sample_indices: np.ndarray, signals: np.ndarray) -> dict:
        #     signal_min = np.array(np.inf)
        #     signal_max = np.array(-np.inf)
        #     n_bins = 200
        #     cummulative_hist = np.zeros(n_bins, dtype=int)
        #     stat_samples = int(10e3)
        #     stat_interval = np.max([1, sample_indices.size//stat_samples])
        #     for i in sample_indices[::stat_interval]:
        #         signal_window = signals[i: i + self.signal_window_size, :, :]
        #         signal_min = np.min([signal_min, signal_window.min()])
        #         signal_max = np.max([signal_max, signal_window.max()])
        #         hist, bin_edges = np.histogram(
        #             signal_window,
        #             bins=n_bins,
        #             range=[-10.4, 10.4],
        #         )
        #         cummulative_hist += hist
        #     bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        #     mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
        #     stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
        #     exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
        #     self.zprint(f"    Stats: mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} min/max {signal_min:.3f}/{signal_max:.3f}")
        #     return {
        #         'count': sample_indices.size,
        #         'min': signal_min,
        #         'max': signal_max,
        #         'mean': mean,
        #         'stdev': stdev,
        #         'exkurt': exkurt,
        #     }

        # # valid t0 indices
        # packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        # packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        # assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        # self.zprint(f"  Stage {stage.upper()}: Raw data stats")
        # # assert packaged_labels.size == packaged_signals.shape[0]

        # # standardize signals based on training data
        # stats = _get_statistics2(
        #     sample_indices=packaged_valid_t0_indices,
        #     signals=packaged_signals,
        # )
        # if self.is_global_zero and stage == 'train' and not self.confinement_raw_signal_mean:
        #     self.zprint(f"    Using {stage.upper()} for standarizing mean and stdev")
        #     self.confinement_raw_signal_mean = stats['mean']
        #     self.confinement_raw_signal_stdev = stats['stdev']
        #     self.save_hyperparameters({
        #         'signal_mean': self.confinement_raw_signal_mean.item(),
        #         'signal_stdev': self.confinement_raw_signal_stdev.item(),
        #     })
        # self.barrier()
        # self.confinement_raw_signal_mean = self.broadcast(self.confinement_raw_signal_mean)
        # self.confinement_raw_signal_stdev = self.broadcast(self.confinement_raw_signal_stdev)
        # self.rprint(f"  Stage {stage.upper()}: Standarizing signals with mean {self.confinement_raw_signal_mean:.3f} and std {self.confinement_raw_signal_stdev:.3f}")
        # packaged_signals = (packaged_signals - self.confinement_raw_signal_mean) / self.confinement_raw_signal_stdev

        # self.rprint(f"  Stage {stage.upper()}: Valid t0 indices: {len(packaged_valid_t0_indices):,d}")
        # self.zprint(f"  Stage {stage.upper()}: Batches per epoch: {len(packaged_valid_t0_indices)/self.batch_size:.1f}")
        # self.zprint(f"  Stage {stage.upper()}: Load/preprocess confinement data time: {time.time()-t_tmp:.1f} s")

        # if stage == 'train':
        #     dataset = Confinement_TrainValTest_Dataset(
        #         signals=packaged_signals,
        #         n_rows=self.n_rows,
        #         n_cols=self.n_cols,
        #         labels=packaged_labels,
        #         sample_indices=packaged_valid_t0_indices,
        #         window_start_indices=packaged_window_start,
        #         signal_window_size=self.signal_window_size,
        #         confinement_mode_keys=packaged_confinement_mode_key,
        #     )
        #     return dataset
        # elif stage in ['validation', 'test']:
        #     self.confinement_datasets[stage] = Confinement_TrainValTest_Dataset(
        #             signals=packaged_signals,
        #             n_rows=self.n_rows,
        #             n_cols=self.n_cols,
        #             labels=packaged_labels,
        #             sample_indices=packaged_valid_t0_indices,
        #             window_start_indices=packaged_window_start,
        #             signal_window_size=self.signal_window_size,
        #             confinement_mode_keys=packaged_confinement_mode_key,
        #         )
        #     return
        # elif stage == 'predict':
        #     del self.confinement_train_dataloader
        #     del self.confinement_datasets['validation']
        # else:
        #     raise ValueError
            
        # gc.collect()
        # torch.cuda.empty_cache()
        # self.rprint(f'The CPU usage is: {psutil.cpu_percent(4)}')
        # self.rprint(f'RAM memory % used: {psutil.virtual_memory()[2]}')
        # self.rprint(f'RAM Used (GB): {psutil.virtual_memory()[3]/1000000000}')    

    def train_dataloader(self) -> dict[str, torch.utils.data.DataLoader]:
        result = {}
        if self.elm_classifier:
            result['elm_classifier'] = self._elm_train_val_test_dataloaders('train')
        if self.conf_classifier:
            result['conf_classifier'] = self.confinement_train_dataloader
        return result

    def val_dataloader(self) -> dict[str, torch.utils.data.DataLoader]:
        result = {}
        if self.elm_classifier:
            result['elm_classifier'] = self._elm_train_val_test_dataloaders('validation')
        if self.conf_classifier:
            sampler = (
                torch.utils.data.DistributedSampler(
                    dataset=self.confinement_datasets['validation'],
                    shuffle=False,
                    drop_last=True,
                )
                if self.trainer.world_size > 1
                else torch.utils.data.SequentialSampler(self.confinement_datasets['validation'])
            )
            confinement_val_dl = torch.utils.data.DataLoader(
                dataset=self.confinement_datasets['validation'],
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                # pin_memory=True,
                persistent_workers=(self.num_workers > 0),
            )
            result['conf_classifier'] = confinement_val_dl
        return result

    def test_dataloader(self) -> dict[str, torch.utils.data.DataLoader]:
        confinement_test_dl = torch.utils.data.DataLoader(
            dataset=self.confinement_datasets['test'],
            sampler=torch.utils.data.DistributedSampler(
                self.confinement_datasets['test'],
                shuffle=False,
                drop_last=True,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        ) 
        return {
            'elm_classifier': self._elm_train_val_test_dataloaders('test'),
            'conf_classifier': confinement_test_dl,
        }

    def predict_dataloader(self) -> None:
        pass

    def _elm_train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        sampler = (
            torch.utils.data.RandomSampler(data_source=self.elm_datasets[stage])
            if stage == 'train'
            else torch.utils.data.SequentialSampler(data_source=self.elm_datasets[stage])
        )
        if self.epochs_per_batch_size_reduction and stage == 'train':
            batch_size_reduction_pow2_factor = min(
                self.max_pow2_batch_size_reduction, 
                self.trainer.current_epoch//self.epochs_per_batch_size_reduction,
            ) 
            batch_size_per_rank = self.batch_size_per_rank // (2**batch_size_reduction_pow2_factor)
            if batch_size_per_rank != self.batch_size_per_rank:
                self.zprint(f"Reduced global batch size: {batch_size_per_rank}")
        else:
            batch_size_per_rank = self.batch_size_per_rank
        return torch.utils.data.DataLoader(
            dataset=self.elm_datasets[stage],
            sampler=sampler,
            batch_size=batch_size_per_rank,  # batch size per rank
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers else None,
            pin_memory=True,
            drop_last=True,
        )

    def get_state_dict(self) -> dict:
        state_dict = {item: getattr(self, item) for item in self.state_items}
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])
            self.zprint(f"Loading state item {item} = {getattr(self, item)}")

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.trainer.world_size > 1:
            self.barrier()
            print(f"Rank {self.trainer.global_rank}: {text}")
            self.barrier()
        else:
            print(text)


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
        return signal_window, quantile_binary_label, time_to_elm


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
        elm_data_file: str|Path = None,
        confinement_data_file: str|Path = None,
        elm_classifier=True,
        conf_classifier=False,
        max_elms: int|Any = None,
        signal_window_size = 1024,
        experiment_name = 'experiment_default',
        # model
        lr = 1e-3,
        weight_decay = 1e-4,
        lr_scheduler_patience = 20,
        lr_warmup_epochs: int = 5,
        monitor_metric = None,
        use_optimizer = 'SGD',
        # loggers
        log_freq = 100,
        use_wandb = False,
        # callbacks
        early_stopping_min_delta = 1e-3,
        early_stopping_patience = 20,
        # trainer
        max_epochs = 2,
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        skip_train: bool = False,
        precision = None,
        # data
        batch_size = 64,
        fraction_validation = 0.15,
        fraction_test = 0.0,
        num_workers = None,
        time_to_elm_quantile_min: float|Any = None,
        time_to_elm_quantile_max: float|Any = None,
        contrastive_learning: bool = True,
        min_pre_elm_time: float|Any = None,
        fir_bp_low = None,
        fir_bp_high = None,
        epochs_per_batch_size_reduction: int = None,
        max_shots_per_class: int = None,
        max_confinement_event_length: int = None,
):

    # SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    global_rank = int(os.getenv("SLURM_PROCID", default=0))
    local_rank = int(os.getenv("SLURM_LOCALID", default=0))
    node_rank = int(os.getenv("SLURM_NODEID", default=0))

    is_global_zero = (global_rank == 0)

    def zprint(text):
        if is_global_zero:
            print(text)

    def rprint(text):
        if world_size==1:
            print(text)
        else:
            print(f"Global rank {global_rank}: {text}")

    ### model
    lit_model = Model(
        elm_classifier=elm_classifier,
        conf_classifier=conf_classifier,
        signal_window_size=signal_window_size,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_warmup_epochs=lr_warmup_epochs,
        weight_decay=weight_decay,
        monitor_metric=monitor_metric,
        use_optimizer=use_optimizer,
        is_global_zero=is_global_zero,
    )
    monitor_metric = lit_model.monitor_metric
    lit_model.save_hyperparameters({
        'gradient_clip_val': gradient_clip_val, 
        'gradient_clip_algorithm': gradient_clip_algorithm, 
        'precision': precision,
    })

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
    zprint(f"Trial directory: {trial_dir}")
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

    zprint(f"World size {world_size} on {num_nodes} node(s)")
    rprint(f"Local rank {local_rank} on node {node_rank}")

    zprint("Model Summary:")
    zprint(ModelSummary(lit_model, max_depth=-1))

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
        # enable_progress_bar = True if __name__=='__main__' else False,
        enable_progress_bar = False,
        enable_model_summary = False,
        precision = precision,
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=True,
        ) if world_size>1 else 'auto',
        num_nodes = num_nodes,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=10,
    )

    assert trainer.node_rank == node_rank
    assert trainer.world_size == world_size
    assert trainer.local_rank == local_rank
    assert trainer.global_rank == global_rank
    assert trainer.is_global_zero == is_global_zero

    ### data
    lit_datamodule = Data(
        signal_window_size=signal_window_size,
        elm_data_file=elm_data_file,
        confinement_data_file=confinement_data_file,
        elm_classifier=lit_model.elm_classifier,
        conf_classifier=lit_model.conf_classifier,
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
        fir_bp_low=fir_bp_low,
        fir_bp_high=fir_bp_high,
        epochs_per_batch_size_reduction=epochs_per_batch_size_reduction,
        max_shots_per_class=max_shots_per_class,
        max_confinement_event_length=max_confinement_event_length,
    )

    if skip_train is False:
        trainer.fit(lit_model, datamodule=lit_datamodule)
        if fraction_test:
            trainer.test(lit_model, lit_datamodule)

    if use_wandb:
        wandb.finish()

if __name__=='__main__':
    main(
        elm_classifier=False,
        conf_classifier=True,
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/labeled_elm_events.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=100,
        batch_size=128,
        lr=1e-3,
        max_epochs=2,
        num_workers=2,
        log_freq=50,
        fraction_validation=0.25,
        fraction_test=0.0,
        time_to_elm_quantile_min=0.4,
        time_to_elm_quantile_max=0.6,
        contrastive_learning=True,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        # max_shots_per_class=6,
        # max_confinement_event_length=int(20e3),
        # use_wandb=True,
    )