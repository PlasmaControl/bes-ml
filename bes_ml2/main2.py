from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
from logging import Logger
from collections.abc import Iterable, Mapping, Callable
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
from lightning.pytorch.utilities.model_summary import ModelSummary


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

    def __post_init__(self):
        assert np.log2(self.signal_window_size).is_integer(), \
            'Signal window must be power of 2'


class _LitWrapper(LightningModule):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model
    
    def forward(self, inputs):
        return self.torch_model(inputs)


@dataclasses.dataclass(eq=False)
class Model(LightningModule, _Base_Class):
    lr: float = 1e-3  # maximum LR used by first layer
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

        print_fields(self)

        # input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.feature_space_model, self.feature_space_size = self.make_feature_model()

        # task sub-models and metrics
        self.task_models: Mapping[str, LightningModule] = torch.nn.ModuleDict()
        self.task_metrics: dict[str, dict] = {}

        # binary classifier task
        task_name = 'classifier'
        self.task_models[task_name] = self.make_mlp_classifier()
        self.task_metrics[task_name] = {
            'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
            'f1_score': sklearn.metrics.f1_score,
        }

        self.is_global_zero: int = None

        self.total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total model parameters: {self.total_parameters:,}")

        self.initialize_parameters()

        print("Example batch evaluation with batch_size=128")
        self.example_batch_data = torch.zeros(
            size=[128]+list(self.input_data_shape[1:]),
            dtype=torch.float32,
        )
        example_batch_output = self(self.example_batch_data)
        for task_name, task_output in example_batch_output.items():
            print(f"  {task_name} output shape: {task_output.shape}")

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_space_model(signals)
        results = {
            task_model_name: task_model(features)
            for task_model_name, task_model in self.task_models.items()
        }
        return results

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # print(f"  train step batch size: {batch[1].numel()} (global rank {self.global_rank})")
        return self.update_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx) -> None:
        # print(f"  val step batch size: {batch[1].numel()} (global rank {self.global_rank})")
        self.update_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='test')

    def update_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        signal_window, time_to_elm, quantiles = batch
        task_results = self(signal_window)
        sum_loss = None
        for task, task_metrics in self.task_metrics.items():
            results: torch.Tensor = task_results[task]
            if task == 'classifier':
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
        if self.trainer.is_global_zero:
            print("Fit start")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()

    def on_train_epoch_end(self):
        delt = time.time() - self.t_train_epoch_start
        if self.is_global_zero and self.global_step > 0:
            logged_metrics = self.trainer.logged_metrics
            line =  f"  Epoch {self.current_epoch} time: {delt/60:.1f} min  " 
            line += f"steps {self.global_step:,d}  "
            line += f"train loss {logged_metrics['sum_loss/train']:.4f}  "
            line += f"val loss {logged_metrics['sum_loss/val']:.4f}  "
            print(line)

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        if self.is_global_zero:
            print(f"Fit time: {delt/60:0.1f} min")

    def make_feature_model(self) -> tuple[LightningModule, int]:

        print("Feature space sub-model")

        feature_model = torch.nn.Sequential()

        conv_layers = {
            'conv_time_0':  {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            'conv_space_1': {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            'conv_time_2':  {'out_channels': 4, 'kernel': (4, 1, 1), 'stride': (4, 1, 1)},
            'conv_space_3': {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            'conv_space_5': {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
            'conv_space_6': {'out_channels': 4, 'kernel': (1, 2, 2), 'stride': 1},
        }

        data_shape = self.input_data_shape
        print(f"  Data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels = 1
        for layer_name, layer in conv_layers.items():
            conv = torch.nn.Conv3d(
                in_channels=out_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
            )
            n_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            print(f"  {layer_name} kern: {conv.kernel_size} stride: {conv.stride} out_ch: {conv.out_channels} param: {n_params:,d} output: {data_shape} (size {np.prod(data_shape)})")
            out_channels = conv.out_channels
            feature_model.append(conv)
            feature_model.append(torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope))

        feature_model.append(torch.nn.Flatten())
        output_shape = tuple(feature_model(torch.zeros(self.input_data_shape)).shape)
        print(f"  Flattened feature space shape: {output_shape}")
        n_params = sum(p.numel() for p in feature_model.parameters() if p.requires_grad)
        print(f"  Feature sub-model parameters: {n_params:,d}")

        return _LitWrapper(feature_model), np.prod(output_shape)

    def make_mlp_classifier(self) -> LightningModule:

        print("MLP classifier sub-model")

        mlp_classifier = torch.nn.Sequential()

        assert self.feature_space_size
        mlp_layers = (self.feature_space_size, 64, 32, 1)

        for i in range(len(mlp_layers)-1):
            fc_layer = torch.nn.Linear(
                in_features=mlp_layers[i],
                out_features=mlp_layers[i+1],
            )
            n_params = sum(p.numel() for p in fc_layer.parameters() if p.requires_grad)
            print(f"  Fully connected layer {i+1} in_features {fc_layer.in_features} out_features {fc_layer.out_features} parameters: {n_params:,d}")
            mlp_classifier.append(fc_layer)
            mlp_classifier.append(torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope))

        n_params = n_params = sum(p.numel() for p in mlp_classifier.parameters() if p.requires_grad)
        print(f"  MLP sub-model parameters: {n_params:,d}")

        return _LitWrapper(mlp_classifier)

    def initialize_parameters(self):
        print("Initializing model to uniform random weights and biases=0")
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                print(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith(".weight"):
                n_in = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / n_in)
                param.data.uniform_(-sqrt_k, sqrt_k)
                print(f"  {name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")

    def setup(self, stage=None):  # fit, validate, test, or predict
        self.is_global_zero = self.trainer.is_global_zero

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
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'monitor': self.monitor_metric,
            },
        }


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signals: np.ndarray = None
    t0_and_time_to_elm_labels: dict[int, float] = None
    signal_window_size: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.signals = torch.from_numpy(self.signals[np.newaxis, ...])
        self.time_to_elm_labels = [val for val in self.t0_and_time_to_elm_labels.values()]
        self.t0_indices = [key for key in self.t0_and_time_to_elm_labels.keys()]
        quantiles = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        self.quantiles = {
            q: qval 
            for q, qval in zip(quantiles, np.quantile(self.time_to_elm_labels, quantiles))
        }

    def __len__(self) -> int:
        return len(self.t0_and_time_to_elm_labels)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i_t0 = self.t0_indices[i]
        time_to_elm = self.time_to_elm_labels[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        quantile_binary_label = {q: int(time_to_elm<=qval) for q, qval in self.quantiles.items()}
        return signal_window, time_to_elm, quantile_binary_label

@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    data_file: str|Path = None
    max_elms: int = None
    batch_size_per_rank: int = 128
    stride_factor: int = 8
    num_workers: int = 0
    outlier_value: float = 6
    fraction_validation: float = 0.2
    fraction_test: float = 0.2
    use_random_data: bool = False
    seed: int = 0  # seed for ELM index shuffling; must be same across processes
    # is_distributed: bool = False

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.data_file = Path(self.data_file).absolute()

        self.datasets: dict[str, ELM_TrainValTest_Dataset] = {}
        self.elm_indices: dict[str,Iterable] = {cat: None for cat in ['all','train','validation','test']}
        self.train_quantiles: dict[float, int] = {}

        self.is_distributed = None

        print_fields(self)

        # datamodule state, to reproduce pre-processing
        self.state_items = [
            'raw_signal_mean',
            'raw_signal_stdev',
        ]
        for item in self.state_items:
            if not hasattr(self, item):
                setattr(self, item, None)

    def get_state_dict(self) -> dict:
        state_dict = {}
        for item in self.state_items:
            state_dict[item] = getattr(self, item)
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])

    def setup(self, stage: str):
        self.is_distributed = self.trainer.world_size > 1
        print(f"Batch size per rank: {self.batch_size_per_rank}  (world size {self.trainer.world_size})")
        assert stage in ['fit', 'test','predict']
        if self.elm_indices['all'] is None:
            self._get_elm_indices_and_split()

        stages = ['train', 'validation'] if stage == 'fit' else [stage.value]
        for st in stages:
            assert st in ['train', 'validation','test','predict']
            if st in self.datasets and isinstance(self.datasets[st], torch.utils.data.Dataset):
                continue
            assert self.elm_indices[st] is not None
            indices = self.elm_indices[st]
            n_indices = len(indices)
            print(f"Reading {n_indices} ELMs for stage {st}")
            elm_data = []
            with h5py.File(self.data_file, 'r') as h5_file:
                elms = h5_file['elms']
                for i_elm, elm_index in enumerate(indices):
                    if i_elm%100 == 0:
                        print(f"  Reading ELM event {i_elm:04d}/{n_indices:04d}")
                    elm_event = elms[f"{elm_index:06d}"]
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                    time = np.array(elm_event['bes_time'], dtype=np.float32)
                    assert time.size == signals.shape[0]
                    t_start = elm_event.attrs['t_start']
                    t_stop = elm_event.attrs['t_stop'] - 0.05
                    t_mask = (time >= t_start) & (time <= t_stop)
                    signals = signals[t_mask, ...]
                    time_to_elm = (time[t_mask] - time[t_mask][-1]) * -1
                    valid_t0 = np.zeros(time_to_elm.size, dtype=int)
                    s_end = len(time_to_elm)
                    while True:
                        s_start = s_end - self.signal_window_size
                        if s_start < 0: break
                        valid_t0[s_start] = 1
                        s_end -= self.signal_window_size // self.stride_factor
                    assert signals.shape[0] == time_to_elm.size
                    assert time_to_elm.size == valid_t0.size
                    elm_data.append({
                        'signals': signals,
                        'time_to_elm': time_to_elm,
                        'valid_t0': valid_t0,
                        'elm_index': elm_index,
                        'shot': elm_event.attrs['shot'],
                        'time_t0': time[t_mask][0],
                    })
            
            concat_signals = np.concatenate(
                [elm['signals'] for elm in elm_data],
                axis=0,
            )
            concat_time_to_elm = np.concatenate(
                [elm['time_to_elm'] for elm in elm_data],
            )
            concat_valid_t0 = np.concatenate(
                [elm['valid_t0'] for elm in elm_data],
            )

            t0_and_time_to_elm_labels = {
                i_t0: concat_time_to_elm[i_t0 + self.signal_window_size - 1]
                for i_t0, is_valid_t0 in enumerate(concat_valid_t0) if is_valid_t0
            }

            # remove signal windows with outliers
            if self.outlier_value:
                outlier_count = 0
                for i_t0 in list(t0_and_time_to_elm_labels.keys()):
                    signal_window = concat_signals[..., i_t0 : i_t0 + self.signal_window_size, :, :]
                    if np.abs(signal_window).max() > self.outlier_value:
                        del t0_and_time_to_elm_labels[i_t0]
                        outlier_count += 1
                print(f"  Outlier signal windows removed: {outlier_count:,d}")
            
            # Window and batch counts
            window_count = len(t0_and_time_to_elm_labels)
            total_batches = window_count / self.batch_size_per_rank
            batches_per_rank = total_batches / self.trainer.world_size
            print(f"  Signal window count: {window_count:,d}  Batches: {total_batches:,.1f}  Batches/rank: {batches_per_rank:,.1f}")

            # Raw signal stats
            raw_stats = self._get_statistics(
                signals=concat_signals,
                sample_indices=np.array(list(t0_and_time_to_elm_labels.keys()), dtype=int),
            )
            print(f"  Raw signals min {raw_stats['min']:.2f} max {raw_stats['max']:.2f} mean {raw_stats['mean']:.2f} stdev {raw_stats['stdev']:.2f} exkurt {raw_stats['exkurt']:.2f}")
            if st == 'train':
                self.raw_signal_mean = raw_stats['mean']
                self.raw_signal_stdev = raw_stats['stdev']
            else:
                assert self.raw_signal_mean and self.raw_signal_stdev

            # normalize signals
            concat_signals = (concat_signals-self.raw_signal_mean) / self.raw_signal_stdev
            norm_stats = self._get_statistics(
                signals=concat_signals,
                sample_indices=np.array(list(t0_and_time_to_elm_labels.keys()), dtype=int),
            )
            print(f"  Normalized signals min {norm_stats['min']:.2f} max {norm_stats['max']:.2f} mean {norm_stats['mean']:.2f} stdev {norm_stats['stdev']:.2f} exkurt {norm_stats['exkurt']:.2f}")

            # create datasets
            if st in ['train', 'validation', 'test']:
                self.datasets[st] = ELM_TrainValTest_Dataset(
                    signals=concat_signals,
                    t0_and_time_to_elm_labels=t0_and_time_to_elm_labels,
                    signal_window_size=self.signal_window_size,
                )
                if st == 'train':
                    self.train_quantiles = self.datasets[st].quantiles
            
            if st in ['test', 'predict']:
                pass

    def _get_elm_indices_and_split(self):
        with h5py.File(self.data_file, 'r') as root:
            shots = [int(shot_key) for shot_key in root['shots']]
            assert len(shots) == len(set(shots))
            shots = set(shots)
            shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(shots ^ shots_from_elms) == 0
            elms = [int(elm_key) for elm_key in root['elms']]
        # shuffle ELM indices
        print(f"Total ELMs in dataset: {len(elms)}")
        print(f"Total shots in dataset: {len(shots)}")
        print(f"Shuffling ELMs with seed={self.seed}")
        np.random.default_rng(self.seed).shuffle(elms)
        # limit number of ELM events
        if self.max_elms:
            elms = elms[:self.max_elms]
        # split ELM indicies
        self.elm_indices['all'] = tuple(elms)
        n_elms = len(self.elm_indices['all'])
        n_test_elms = int(self.fraction_test * n_elms)
        n_validation_elms = int(self.fraction_validation * n_elms)
        self.elm_indices['test'] = tuple(self.elm_indices['all'][:n_test_elms])
        train_val_elm_indices = self.elm_indices['all'][n_test_elms:]
        self.elm_indices['validation'] = train_val_elm_indices[:n_validation_elms]
        self.elm_indices['train'] = train_val_elm_indices[n_validation_elms:]
        print("ELMs for analysis")
        for stage, elm_indices in self.elm_indices.items():
            tmp = f"  {stage.capitalize()} ELMs: {len(elm_indices)}"
            if stage != 'all':
                tmp += f" ({len(elm_indices)/len(self.elm_indices['all'])*1e2:.1f}%)"
            print(tmp)

    def _train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        shuffle = True if stage=='train' else False
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.datasets[stage],
            shuffle=shuffle,
        ) if self.is_distributed else None
        return torch.utils.data.DataLoader(
            dataset=self.datasets[stage],
            sampler=sampler,
            batch_size=self.batch_size_per_rank,
            num_workers=self.num_workers,
            shuffle=None if self.is_distributed else shuffle,
            prefetch_factor=2 if self.num_workers else None,
            persistent_workers=bool(self.num_workers),
            # pin_memory=False,
            # pin_memory_device="",
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('train')

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('validation')

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_val_test_dataloaders('test')

    def predict_dataloader(self) -> list[torch.utils.data.DataLoader]:
        return [
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size_per_rank,
                num_workers=self.num_workers,
                persistent_workers=True,
            ) for dataset in self.datasets['predict']
        ]

    def _get_statistics(
            self, 
            signals: np.ndarray,
            sample_indices: np.ndarray, 
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
        return {
            'count': sample_indices.size,
            'min': signal_min,
            'max': signal_max,
            'mean': mean,
            'stdev': stdev,
            'exkurt': exkurt,
        }



if __name__=='__main__':

    world_size = int(os.getenv('WORLD_SIZE', default=0))
    # world_size = 0
    batch_size_per_rank = 32
    signal_window_size = 1024
    max_epochs = 8
    max_steps = -1
    max_elms = 20
    fraction_test = 0
    lr = 5e-3
    log_freq = 10
    early_stopping_min_delta = 1e-3
    early_stopping_patience = 5
    use_wandb = False

    experiment_name = 'experiment_default'
    experiment_dir = Path(experiment_name).absolute()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
    trial_name = f"r{slurm_identifier}_{datetime_str}" if slurm_identifier else f"r{datetime_str}"

    torch.set_default_dtype(torch.float32)

    ### model
    lit_model = Model(
        signal_window_size=signal_window_size,
        lr=lr,
    )
    # test_output = lit_model(lit_model.example_batch_data)
    monitor_metric = lit_model.monitor_metric
    metric_mode = 'min' if 'loss' in monitor_metric else 'max'

    print("Model Summary:")
    print(ModelSummary(lit_model, max_depth=-1))

    ### data
    lit_datamodule = Data(
        signal_window_size = signal_window_size,
        data_file = '/Users/drsmith/Documents/repos/bes-ml/bes_ml2/small_elm_data.hdf5',
        max_elms= max_elms,
        batch_size_per_rank = batch_size_per_rank,
        fraction_test=fraction_test,
        num_workers=2,
    )

    ### loggers
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir.parent,
        name=experiment_name,
        version=trial_name,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    trial_dir = Path(tb_logger.log_dir).absolute()
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

    ### callbacks
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor=monitor_metric,
            mode=metric_mode,
            save_last=True,
        ),
        DeviceStatsMonitor(),
        # EarlyStopping(
        #     monitor=monitor_metric,
        #     mode=metric_mode,
        #     min_delta=early_stopping_min_delta,
        #     patience=early_stopping_patience,
        #     log_rank_zero_only=True,
        #     verbose=True,
        # ),
    ]
    ### initialize trainer
    trainer = Trainer(
        max_epochs = max_epochs,
        max_steps = max_steps,
        max_time = None,
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
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
        ) if world_size else 'auto',
        use_distributed_sampler = bool(world_size),
        devices = world_size if world_size else "auto",
        num_nodes = int(os.getenv('SLURM_NNODES', default=1)),
    )

    trainer.fit(lit_model, datamodule=lit_datamodule)

    if fraction_test:
        trainer.test(lit_model, lit_datamodule)

    if use_wandb:
        wandb.finish()