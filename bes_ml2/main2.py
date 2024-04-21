from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
from logging import Logger
from collections.abc import Iterable, Mapping, Callable
import os


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
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint


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
    log_dir: str = dataclasses.field(default='.', init=False)

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
        self.task_metrics: dict = {}

        # binary classifier task
        task_name = 'classifier'
        self.task_models[task_name] = self.make_mlp_classifier()
        self.task_metrics[task_name] = {
            'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
            'f1_score': sklearn.metrics.f1_score,
        }

        self.total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total model parameters: {self.total_parameters:,}")

        # self.initialize_parameters()

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
        return self.update_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx, stage='test')

    def update_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        signals, labels = batch
        results = self(signals)
        sum_loss = None
        for metric_name, func in self.metric_functions.items():
            if 'loss' in metric_name:
                metric_value = func(
                    input=results,
                    target=labels.type_as(results),
                )
                sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
            elif 'score' in metric_name:
                kwargs = {}
                if 'f1' in metric_name:
                    modified_predictions = (results > 0.5).type(torch.int)
                    kwargs['zero_division'] = 0
                else:
                    modified_predictions = results
                metric_value = func(
                    y_pred=modified_predictions.detach().cpu(), 
                    y_true=labels.detach().cpu(),
                    **kwargs,
                )
            self.log(f"{metric_name}/{stage}", metric_value, sync_dist=True)
        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)
        return sum_loss

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
        print("Initializing model parameters")
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                print(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith(".weight"):
                n_in = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / n_in)
                param.data.uniform_(-sqrt_k, sqrt_k)
                print(f"  {name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")

    def setup(
            self, 
            stage: str = None,  # fit, validate, test, or predict
    ):
        # called in every process at beginning of every stage
        pass
        # datamodule = self.trainer.datamodule
        # for label_percentile in ['label_scaled_25p', 'label_scaled_50p', 'label_scaled_75p']:
        #     assert hasattr(datamodule, label_percentile)
        #     setattr(self, label_percentile, getattr(datamodule, label_percentile))

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
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'monitor': self.monitor_metric,
            },
        }


# @dataclasses.dataclass(eq=False)
# class Random_Dataset(torch.utils.data.IterableDataset, _Base_Class):
#     dataset_size: int = int(1e4)

#     def __post_init__(self):
#         super().__init__()
#         super(torch.utils.data.IterableDataset, self).__post_init__()
#         self.samples = []
#         for _ in range(self.dataset_size):
#             self.samples.append((
#                 torch.randn((1, self.signal_window_size, 8, 8)),
#                 torch.rand((1)),
#                 torch.randint(0, 2, (1)),
#             ))

#     def __iter__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         return iter(self.samples)

@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signals: np.ndarray = None
    labels: np.ndarray = None
    sample_indices: np.ndarray = None
    signal_window_size: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()

    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_window = None
        label_time_to_elm = None
        label_50p = None
        return signal_window, label_time_to_elm, label_50p

@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    data_file: str|Path = None
    max_elms: int = None
    batch_size_per_worker: int = 128
    num_workers: int = 4
    fraction_validation: float = 0.2
    fraction_test: float = 0.2
    use_random_data: bool = False
    seed: int = 0  # seed for ELM index shuffling; must be same across processes

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.data_file = Path(self.data_file).absolute()

        self.datasets = {}
        self.elm_indices: dict[str,Iterable] = {cat: None for cat in ['all','train','validation','test']}
        # self.shots: dict[str,Iterable] = {cat: None for cat in ['all','train','validation','test']}

        print_fields(self)

        self.state_items = []

    def get_state_dict(self) -> dict:
        state_dict = {}
        for item in self.state_items:
            state_dict[item] = getattr(self, item)
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])

    def setup(self, stage: str):
        if self.elm_indices['all'] is None:
            self._get_elm_indices_and_split()

        stages = ['train', 'validation'] if stage == 'fit' else [stage]
        for st in stages:
            if st in self.datasets and isinstance(self.datasets[st], torch.utils.data.Dataset):
                continue
            assert self.elm_indices[st] is not None
            indices = self.elm_indices[st]
            n_indices = len(indices)
            print(f"Reading {n_indices} ELMs for stage {st}")
            # if self.use_random_data:
            #     self.datasets[st] = Random_Dataset()
            #     continue
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
                    t_start = elm_event.attrs['t_start']
                    t_stop = elm_event.attrs['t_stop']
                    # elm_data.append({})
            
            packaged_labels = []
            packaged_signals = []
            packaged_valid_t0 = []

            packaged_valid_t0_indices = []

            if stage in ['train', 'validation', 'test']:
                self.datasets[stage] = ELM_TrainValTest_Dataset(
                    signals=packaged_signals,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    signal_window_size=self.signal_window_size,
                    # label_scaled_25p=self.label_scaled_25p,
                    # label_scaled_75p=self.label_scaled_75p,
                )
            
            if stage in ['test', 'predict']:
                pass
                # self.datasets['predict'] = ELM_Predict_Dataset()

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
        shuffle = drop_last = True if stage=='train' else False
        return torch.utils.data.DataLoader(
            dataset=self.datasets[stage],
            sampler=torch.utils.data.DistributedSampler(
                dataset=self.datasets[stage],
                shuffle=shuffle,
                drop_last=drop_last,
            ),
            batch_size=self.batch_size_per_worker,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
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
                batch_size=256,
                num_workers=self.num_workers,
                persistent_workers=True,
            ) for dataset in self.datasets['predict']
        ]


if __name__=='__main__':

    ### controls
    signal_window_size = 1024
    max_epochs = 2
    max_steps = 100
    log_freq = 100
    experiment_dir = Path('./experiment_default').absolute()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name
    experiment_parent_dir = experiment_dir.parent
    trial_name = None
    early_stopping_min_delta = 1e-3
    early_stopping_patience = 5

    torch.set_default_dtype(torch.float32)

    ### model
    # model = Model(
    #     signal_window_size=signal_window_size,
    #     lr=1e-3,
    # )
    # test_output = model(model.example_batch_data)
    # monitor_metric = model.monitor_metric
    # metric_mode = 'min' if 'loss' in monitor_metric else 'max'

    ### loggers
    # tb_logger = TensorBoardLogger(
    #     save_dir=experiment_parent_dir,
    #     name=experiment_name,
    #     version=trial_name,
    #     default_hp_metric=False,
    # )
    # trial_dir = Path(tb_logger.log_dir).absolute()
    # trial_dir.mkdir(parents=True, exist_ok=True)
    # print(f"Trial directory: {trial_dir}")
    # wandb.login()
    # wandb_logger = WandbLogger(
    #     save_dir=experiment_dir,
    #     project=experiment_name,
    #     name=trial_name,
    # )
    # wandb_logger.watch(
    #     model=model, 
    #     log='all', 
    #     log_freq=log_freq,
    # )
    loggers = [
        # tb_logger, 
        # wandb_logger,
    ]

    ### callbacks
    callbacks = [
        # LearningRateMonitor(
        #     logging_interval = None,
        #     log_momentum = False,
        #     log_weight_decay = False
        # ),
        # ModelCheckpoint(
        #     monitor=monitor_metric,
        #     mode=metric_mode,
        #     save_last=None,
        # ),
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
    # trainer = Trainer(
    #     gradient_clip_val = None,
    #     gradient_clip_algorithm = None,
    #     max_epochs = max_epochs,
    #     max_steps = max_steps,
    #     max_time = None,
    #     logger = loggers,
    #     callbacks = callbacks,
    #     enable_checkpointing = True,
    #     enable_progress_bar = True,
    #     enable_model_summary = True,
    #     precision = None,
    #     strategy = "auto",
    #     log_every_n_steps = log_freq,
    #     use_distributed_sampler = True,
    #     num_nodes = int(os.getenv('SLURM_NNODES', default=1)),
    # )
    ### data
    data = Data(
        signal_window_size = signal_window_size,
        data_file = '/Users/drsmith/Documents/repos/bes-ml/bes_ml2/small_elm_data.hdf5',
        max_elms= 20,
        batch_size_per_worker = 128,
        fraction_test=0,
        num_workers=2,
    )
    data.setup('fit')
    ### run trainer
    # assert model.signal_window_size == data.signal_window_size
    # trainer.fit(
    #     model=model,
    #     datamodule=data,
    # )

    # wandb.finish()