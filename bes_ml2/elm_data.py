from __future__ import annotations
import dataclasses
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn
import torch.utils.data
import pytorch_lightning as pl

from bes_data.sample_data import sample_elm_data_file
from bes_data.elm_data_tools import bad_elm_indices_csv


class ELM_TrainValTest_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            labels: np.ndarray,
            sample_indices: np.ndarray,
            window_start_indices: np.ndarray,
            signal_window_size: int,
            prediction_horizon: int = 0,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = torch.from_numpy(signals[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == 8 and
            self.signals.size(3) == 8
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        assert self.labels.ndim == 1, "Labels have incorrect shape"
        assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
        self.signal_window_size = signal_window_size
        self.prediction_horizon = prediction_horizon
        self.window_start_indices = torch.from_numpy(window_start_indices)
        self.sample_indices = torch.from_numpy(sample_indices)
        assert torch.max(self.sample_indices)+self.signal_window_size+self.prediction_horizon-1 < self.labels.numel()

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.sample_indices[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        label_index = i_t0 + self.signal_window_size + self.prediction_horizon - 1
        label = self.labels[ label_index : label_index + 1 ]
        return signal_window, label


class ELM_Predict_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            labels: np.ndarray,
            signal_window_size: int,
            prediction_horizon: int = 0,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = torch.from_numpy(signals[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == 8 and
            self.signals.size(3) == 8
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        self.active_elm_start_index = np.flatnonzero(np.isnan(labels))[0]
        assert self.labels.ndim == 1, "Labels have incorrect shape"
        assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
        self.signal_window_size = signal_window_size
        self.prediction_horizon = prediction_horizon
        last_signal_window_start_index = self.labels.numel() - self.signal_window_size
        assert last_signal_window_start_index+self.signal_window_size == self.labels.numel()
        valid_t0 = np.zeros(self.labels.numel(), dtype=int)  # size = n_pre_elm_phase
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
        sample_indices = np.arange(valid_t0.size, dtype=int)
        sample_indices = sample_indices[valid_t0 == 1]
        assert sample_indices.size-1 + self.signal_window_size - 1 == labels.size - 1
        assert sample_indices[-1] + self.signal_window_size-1 == labels.size-1
        self.sample_indices = torch.from_numpy(sample_indices)

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.sample_indices[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        label_index = i_t0 + self.signal_window_size + self.prediction_horizon - 1
        label = self.labels[ label_index : label_index + 1 ]
        return signal_window, label


@dataclasses.dataclass(eq=False)
class ELM_Datamodule_Dataclass():
    data_file: Path|str = sample_elm_data_file.as_posix()  # path to data; dir or file depending on task
    batch_size: int = 128  # power of 2, like 32-256
    signal_window_size: int = 128  # power of 2, like 64-512
    num_workers: int = 4  # number of subprocess workers for pytorch dataloader
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    seed: int = 0  # RNG seed for deterministic, reproducible shuffling of ELM events
    max_elms: int = None
    signal_mean: float = None
    signal_stdev: float = None
    label_median: float = None
    # standardize_signals: bool = True  # if True, standardize signals based on training data mean~0, stdev~1
    clip_sigma_outliers: float = 6.0  # remove signal windows with abs(standardized_signals) > n_sigma
    # standardize_fft: bool = True  # if True, standardize FFTs based on training data log10(FFT^2) mean~0, stdev~1
    bad_elm_indices: list = None  # iterable of ELM indices to skip when reading data
    bad_elm_indices_csv: str | bool = True  # CSV file to read bad ELM indices
    # pre_elm_size: int = None  # maximum pre-ELM window in time frames
    log_time: bool = False  # if True, use label = log(time_to_elm_onset)
    max_predict_elms: int = 12
    prepare_data_per_node: bool = None  # hack to avoid error between dataclass and LightningDataModule


pl.LightningDataModule()
@dataclasses.dataclass(init=True, eq=False)
class ELM_Datamodule(
        ELM_Datamodule_Dataclass,
        pl.LightningDataModule,
    ):

    def __post_init__(self):
        super(ELM_Datamodule_Dataclass).__init__()
        self.save_hyperparameters(ignore=[
            'max_predict_elms',
            'prepare_data_per_node',
        ])
        self.datasets = {}
        self.all_elm_indices = None
        self.test_elm_indices = None
        self.train_elm_indices = None
        self.validation_elm_indices = None

    def prepare_data(self):
        # only called in main process
        pass

    def _get_elm_indices_and_split(self):
        if self.all_elm_indices is not None:
            print("Reusing previous ELM indices read and split")
            return
        print(f"Data file: {self.data_file}")
        # gather ELM indices
        with h5py.File(self.data_file, "r") as elm_h5:
            print(f"  ELM events in data file: {len(elm_h5)}")
            self.all_elm_indices = [int(elm_key) for elm_key in elm_h5]
        # bad ELM events to ignore?
        if self.bad_elm_indices or self.bad_elm_indices_csv:
            if self.bad_elm_indices is None:
                self.bad_elm_indices = []
            if self.bad_elm_indices_csv is True:
                self.bad_elm_indices_csv = bad_elm_indices_csv
            if self.bad_elm_indices_csv:
                print(f"  Ignoring ELM events from {self.bad_elm_indices_csv}")
                with Path(self.bad_elm_indices_csv).open() as file:
                    self.bad_elm_indices = [int(line) for line in file]
            ignored_elm_count = 0
            for bad_elm_index in self.bad_elm_indices:
                if bad_elm_index in self.all_elm_indices:
                    self.all_elm_indices.remove(bad_elm_index)
                    ignored_elm_count += 1
            print(f"  Ignored ELM events: {ignored_elm_count}")
            print(f"  Usable ELM events: {len(self.all_elm_indices)}")
        self.all_elm_indices = np.array(self.all_elm_indices, dtype=int)
        # shuffle ELM indices
        print(f"  Shuffling ELM events with RNG seed {self.seed}")
        np.random.default_rng(seed=self.seed).shuffle(self.all_elm_indices)
        if self.all_elm_indices.size >= 5:
            print(f"  Initial ELM order after shuffling: {self.all_elm_indices[0:5]}")
        # limit number of ELM events
        if self.max_elms:
            self.all_elm_indices = self.all_elm_indices[:self.max_elms]
            print(f"  Limiting data to {self.max_elms} ELM events")
        # split ELM indicies
        n_test_elms = int(self.fraction_test * self.all_elm_indices.size)
        n_validation_elms = int(self.fraction_validation * self.all_elm_indices.size)
        n_train_elms = self.all_elm_indices.size - n_test_elms - n_validation_elms
        print(f"Total ELM events  {self.all_elm_indices.size}")
        print(f"  Train  {n_train_elms}  ({n_train_elms/self.all_elm_indices.size*100:.1f}%)")
        print(f"  Validation  {n_validation_elms}  ({n_validation_elms/self.all_elm_indices.size*100:.1f}%)")
        print(f"  Test  {n_test_elms}  ({n_test_elms/self.all_elm_indices.size*100:.1f}%)")
        # split into test and train/val
        self.test_elm_indices = self.all_elm_indices[:n_test_elms]
        train_val_elm_indices = self.all_elm_indices[n_test_elms:]
        self.validation_elm_indices = train_val_elm_indices[:n_validation_elms]
        self.train_elm_indices = train_val_elm_indices[n_validation_elms:]

    def setup(self, stage=None):
        # called on every process
        # open ELM data file, read ELM indices, removed ignored ELM events
        print(f"Running ELM_Datamodule.setup(stage={stage})")
        self._get_elm_indices_and_split()
        assert self.all_elm_indices is not None

        if stage == 'fit':
            dataset_elm_indices = {
                'train': self.train_elm_indices,
                'validation': self.validation_elm_indices,
            }
        elif stage == 'test' or stage == 'predict':
            dataset_elm_indices = {
                stage: self.test_elm_indices,
            }
        else:
            raise ValueError

        for dataset_stage, indices in dataset_elm_indices.items():

            if dataset_stage in self.datasets and self.datasets[dataset_stage]:
                print(f"Dataset for `{dataset_stage}` is loaded, continuing")
                continue

            # package ELM events into pytorch dataset
            print(f"Reading ELM events for dataset `{dataset_stage}`")
            elm_data = []
            with h5py.File(self.data_file, 'r') as h5_file:
                if indices.size >= 5:
                    print(f"  Initial indices: {indices[:5]}")
                for i_elm, elm_index in enumerate(indices):
                    if i_elm%100 == 0:
                        print(f"  Reading ELM event {i_elm:04d}/{indices.size:04d}")
                    elm_event = h5_file[f"{elm_index:05d}"]
                    signals = np.array(elm_event["signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (<time>, 8, 8)
                    labels = np.array(elm_event["labels"], dtype=int)
                    labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
                    elm_data.append({
                        'signals': signals, 
                        'labels': labels, 
                        'valid_t0': valid_t0,
                    })

            packaged_labels = np.concatenate([elm['labels'] for elm in elm_data], axis=0)
            packaged_signals = np.concatenate([elm['signals'] for elm in elm_data], axis=0)
            packaged_valid_t0 = np.concatenate([elm['valid_t0'] for elm in elm_data], axis=0)
            assert packaged_labels.size == packaged_valid_t0.size
            # start indices for each ELM event in concatenated dataset
            packaged_window_start = []
            index = 0
            for elm in elm_data:
                packaged_window_start.append(index)
                index += elm['labels'].size
            packaged_window_start = np.array(packaged_window_start, dtype=int)
            # valid t0 indices
            packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
            packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
            assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
            assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices + self.signal_window_size]))
            # standardize signals based on training data
            if None in [self.signal_mean, self.signal_stdev]:
                assert dataset_stage == 'train'
                print("  Calculating signal mean and std from training data")
                stats = self._get_statistics(
                    sample_indices=packaged_valid_t0_indices,
                    signals=packaged_signals,
                )
                self.signal_mean = stats['mean']
                self.signal_stdev = stats['stdev']
            print(f"  Standarizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f} form training data")
            packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev
            self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            # clip outlier signals
            if self.clip_sigma_outliers:
                print(f"  Removing {self.clip_sigma_outliers:.1f} sigma outliers from signals")
                mask = np.zeros(packaged_valid_t0_indices.size, dtype=bool)
                for i_t0_index, t0_index in enumerate(packaged_valid_t0_indices):
                    signal_window = packaged_signals[t0_index: t0_index + self.signal_window_size, :, :]
                    mask[i_t0_index] = np.max(np.abs(signal_window)) <= self.clip_sigma_outliers
                packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
                self._get_statistics(
                    sample_indices=packaged_valid_t0_indices,
                    signals=packaged_signals,
                )
            # normalize labels with min=-1 and median=0
            if self.label_median is None:
                assert dataset_stage == 'train'
                print("  Calculating median label from training labels")
                self.label_median = np.median(packaged_labels[packaged_valid_t0_indices+self.signal_window_size])
            print(f"  Normalizing labels (min=-1 and median~0) with median {self.label_median:.1f} form training labels")
            packaged_labels = (packaged_labels - self.label_median) / (self.label_median-1)
            label_min = np.nanmin(packaged_labels)
            label_median = np.median(packaged_labels[packaged_valid_t0_indices+self.signal_window_size])
            print(f"    Label min {label_min:.1f} median {label_median:.1f}")
            assert label_min == -1
            if dataset_stage in ['train', 'validation', 'test']:
                self.datasets[dataset_stage] = ELM_TrainValTest_Dataset(
                    signals=packaged_signals,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    window_start_indices=packaged_window_start,
                    signal_window_size=self.signal_window_size,
                )
            if dataset_stage in ['test', 'predict']:
                predict_datasets = []
                for i_elm, idx_start in enumerate(packaged_window_start):
                    if i_elm == packaged_window_start.size - 1:
                        idx_stop = packaged_labels.size - 1
                    else:
                        idx_stop = packaged_window_start[i_elm+1]-1
                    signals = packaged_signals[idx_start:idx_stop, ...]
                    labels = packaged_labels[idx_start:idx_stop]
                    dataset = ELM_Predict_Dataset(
                        signals=signals,
                        labels=labels,
                        signal_window_size=self.signal_window_size,
                    )
                    predict_datasets.append(dataset)
                self.datasets['predict'] = predict_datasets

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # determine valid t0 indices (start of signal windows) for each ELM event
        # input labels are binary active/inactive ELM labels
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]  # first active ELM index
        assert labels[active_elm_start_index-1] == 0  # last pre-ELM label
        assert labels[active_elm_start_index] == 1  # first active ELM label
        valid_t0 = np.zeros(labels.size, dtype=int)  # size = n_pre_elm_phase
        last_signal_window_start_index = active_elm_start_index - self.signal_window_size - 1
        assert labels[last_signal_window_start_index+self.signal_window_size] == 0
        assert labels[last_signal_window_start_index+self.signal_window_size+1] == 1
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
        # transform labels to time-to-ELM
        labels = np.zeros(labels.size, dtype=np.float32)
        labels[0:active_elm_start_index] = np.arange(active_elm_start_index, 0, -1)
        labels[active_elm_start_index:] = np.nan
        assert np.nanmin(labels) == 1
        assert np.nanmax(labels) == active_elm_start_index
        assert np.all(labels[np.isfinite(labels)]>0)
        valid_labels = labels[valid_t0==1]
        assert np.all(np.isfinite(valid_labels))
        assert np.all(valid_labels>0)
        assert signals.shape[0] == labels.size
        assert signals.shape[0] == valid_t0.size
        if self.log_time:
            labels = np.log10(labels)
        return labels, signals, valid_t0

    def _get_statistics(
            self, 
            sample_indices: np.ndarray, 
            signals: np.ndarray,
    ) -> dict:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_samples = int(1e4)
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
        print(f"    Stats: count {sample_indices.size:,} min {signal_min:.1f} max {signal_max:.1f} mean {mean:.1f} stdev {stdev:.1f}")
        return {
            'count': sample_indices.size,
            'min': signal_min,
            'max': signal_max,
            'mean': mean,
            'stdev': stdev,
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.datasets["validation"],
            batch_size=1024,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.datasets["test"],
            batch_size=1024,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1024,
                num_workers=self.num_workers,
            ) for dataset in self.datasets['predict']
        ]
