from pathlib import Path
import dataclasses
from typing import Union,Iterable, Tuple
import pickle

import numpy as np
import torch
import torch.utils.data
import h5py
import yaml
import matplotlib.pyplot as plt

from bes_data.sample_data import sample_elm_data_file
from bes_data.elm_data_tools import bad_elm_indices_csv
try:
    from .train_base import _Base_Trainer_Dataclass
    from .utilities import merge_pdfs
    from .models import _Base_Features_Dataclass
except ImportError:
    from bes_ml.base.train_base import _Base_Trainer_Dataclass
    from bes_ml.base.utilities import merge_pdfs
    from bes_ml.base.models import _Base_Features_Dataclass


@dataclasses.dataclass(eq=False)
class _ELM_Data_Base(
    _Base_Trainer_Dataclass,
    _Base_Features_Dataclass,
):
    data_location: Union[Path,str] = sample_elm_data_file  # path to data; dir or file depending on task
    batch_size: int = 64  # power of 2, like 16-128
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    test_data_file: str = 'test_data.pkl'
    # normalize_signals: bool = True  # if True, normalize BES signals such that max ~= 1
    standardize_signals: bool = True # if True, normalize training data to mean=0, stdev=1
    clip_n_sigma: int = 6  # remove signal windows with abs(standardized_signals) > n_sigma
    seed: int = None  # RNG seed for deterministic, reproducable shuffling (ELMs, sample indices, etc.)
    data_partition_file: str = 'data_partition.yaml'  # data partition for training, valid., and testing
    max_elms: int = None
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    pin_memory: bool = True  # data loader pinned memory
    bad_elm_indices: Iterable = None  # iterable of ELM indices to skip when reading data
    bad_elm_indices_csv: str | bool = None  # CSV file to read bad ELM indices
    label_type: np.int8 | np.float32 = dataclasses.field(default=None, init=False)

    def _prepare_data(self) -> None:

        self.data_location = Path(self.data_location).resolve()
        assert self.data_location.exists(), f"{self.data_location} does not exist"

        self.rng_generator = np.random.default_rng(seed=self.seed)

        if self.is_regression:
            # float labels for regression
            self.label_type = np.float32
        elif self.is_classification:
            # int labels for classification
            self.label_type = np.int8

        if self.device.type == 'cuda':
            self.num_workers = 2
        self.logger.info(f"Subprocess workers per data loader: {self.num_workers}")

        self._get_data()
        # self._validate_data()
        self._make_datasets()
        self._make_data_loaders()

    def _get_data(self) -> None:
        self.logger.info(f"Data file: {self.data_location}")

        with h5py.File(self.data_location, "r") as data_file:
            if self.bad_elm_indices is None:
                self.bad_elm_indices = []
            if self.bad_elm_indices_csv is True:
                self.bad_elm_indices_csv = bad_elm_indices_csv
            if self.bad_elm_indices_csv:
                self.logger.info(f"Reading bad ELM indices from {self.bad_elm_indices_csv}")
                with Path(self.bad_elm_indices_csv).open() as file:
                    self.bad_elm_indices = [int(line) for line in file]
            good_keys = []
            for key in data_file:
                if int(key) not in self.bad_elm_indices:
                    good_keys.append(key)
                else:
                    self.logger.info(f"  Skipping bad ELM index {int(key)}")
            elm_indices = np.array(
                [int(key) for key in good_keys],
                dtype=int,
            )
            time_frames = sum([data_file[key]['time'].shape[0] for key in good_keys])
        self.logger.info(f"ELM events in data file: {elm_indices.size}")
        self.logger.info(f"Total time frames: {time_frames:,}")

        # shuffle ELM events
        self.rng_generator.shuffle(elm_indices)

        if self.max_elms:
            elm_indices = elm_indices[:self.max_elms]
            self.logger.info(f"Limiting data to {self.max_elms} ELM events")

        if elm_indices.size >= 5:
            self.logger.info(f"Initial ELM indices: {elm_indices[0:5]}")

        n_validation_elms = int(self.fraction_validation * elm_indices.size)
        n_test_elms = int(self.fraction_test * elm_indices.size)

        test_elms, validation_elms, training_elms = np.split(
            elm_indices,
            [n_test_elms, n_test_elms+n_validation_elms]
        )

        with (self.output_dir/self.data_partition_file).open('w') as data_partition_file:
            data_partition = {
                'n_elms': elm_indices.size,
                'data_location': self.data_location.as_posix(),
                'training_elms': training_elms.tolist(),
                'validation_elms': validation_elms.tolist(),
                'test_elms': test_elms.tolist(),
            }
            yaml.safe_dump(
                data_partition,
                data_partition_file,
                default_flow_style=False,
                sort_keys=False,
            )

        # TODO: move to classification trainer
        if self.is_regression:
            self.oversample_active_elm = False

        self.train_data = self.validation_data = self.test_data = None

        self.logger.info(f"Training ELM events: {training_elms.size}")
        self.train_data = self._preprocess_data(
            elm_indices=training_elms,
            shuffle_indices=True,
            oversample_active_elm=self.oversample_active_elm,
        )

        if n_validation_elms:
            self.logger.info(f"Validation ELM events: {validation_elms.size}")
            self.validation_data = self._preprocess_data(
                elm_indices=validation_elms,
                save_filename='validation_elms',
            )
        else:
            self.logger.info("Skipping validation data")

        if n_test_elms:
            self.logger.info(f"Test ELM events: {test_elms.size}")
            self.test_data = self._preprocess_data(
                elm_indices=test_elms,
                save_filename='test_elms',
            )
            test_data_file = self.output_dir / self.test_data_file
            self.logger.info(f"Test data file: {test_data_file}")
            with test_data_file.open('wb') as file:
                pickle.dump(
                    {
                        "signals": self.test_data[0],
                        "labels": self.test_data[1],
                        "sample_indices": self.test_data[2],
                        "window_start": self.test_data[3],
                        "elm_indices": self.test_data[4],
                    },
                    file,
                )
            assert test_data_file.exists(), f"{test_data_file} does not exist"
            self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")
        else:
            self.logger.info("Skipping test data")

    def _preprocess_data(
        self,
        elm_indices: Iterable = None,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
        save_filename: str = '',
    ) -> None:
        packaged_signals = None
        packaged_window_start = None
        packaged_valid_t0 = None
        packaged_labels = None
        if save_filename:
            plt.ioff()
            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
            self.logger.info(f"  Plotting valid indices: {save_filename}_**.pdf")
            i_page = 1
        with h5py.File(self.data_location, 'r') as h5_file:
            for i_elm, elm_index in enumerate(elm_indices):
                if save_filename and i_elm%12==0:
                    for axis in axes.flat:
                        plt.sca(axis)
                        plt.cla()
                elm_key = f"{elm_index:05d}"
                elm_event = h5_file[elm_key]
                signals = np.array(elm_event["signals"], dtype=np.float32)  # (64, <time>)
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (<time>, 8, 8)
                try:
                    labels = np.array(elm_event["labels"], dtype=self.label_type)
                except KeyError:
                    labels = np.array(elm_event["manual_labels"], dtype=self.label_type)
                labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
                assert labels.size == valid_t0.size
                if save_filename:
                    plt.sca(axes.flat[i_elm%12])
                    plt.plot(signals[:,2,3]/10, label='BES 20')
                    plt.plot(signals[:,2,5]/10, label='BES 22')
                    plt.plot(labels, label='Label')
                    plt.title(f"ELM index {elm_key}")
                    plt.legend(fontsize='x-small')
                    plt.xlabel('Time (mu-s)')
                    if i_elm%12==11 or i_elm==elm_indices.size-1:
                        plt.tight_layout()
                        output_file = self.output_dir/(save_filename + f"_{i_page:02d}.pdf")
                        plt.savefig(
                            output_file, 
                            format="pdf", 
                            transparent=True,
                        )
                        i_page += 1
                if packaged_signals is None:
                    packaged_window_start = np.array([0])
                    packaged_valid_t0 = valid_t0
                    packaged_signals = signals
                    packaged_labels = labels
                else:
                    last_index = packaged_labels.size - 1
                    packaged_window_start = np.append(
                        packaged_window_start, 
                        last_index + 1
                    )
                    packaged_valid_t0 = np.concatenate([packaged_valid_t0, valid_t0])
                    packaged_signals = np.concatenate([packaged_signals, signals], axis=0)
                    packaged_labels = np.concatenate([packaged_labels, labels], axis=0)

        assert packaged_labels.size == packaged_valid_t0.size

        if save_filename and self.is_main_process:
            plt.close()
            pdf_files = sorted(self.output_dir.glob(f'{save_filename}_*.pdf'))
            output = self.output_dir / f'{save_filename}.pdf'
            merge_pdfs(pdf_files, output, delete_inputs=True)
        
        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        if self.is_classification:
            # assess data balance for active ELM/inactive ELM classification
            packaged_valid_t0_indices = self._check_for_balanced_data(
                packaged_labels=packaged_labels,
                packaged_valid_t0_indices=packaged_valid_t0_indices,
                oversample_active_elm=oversample_active_elm,
            )
        elif self.is_regression:
            # if specified, normalize time-to-ELM labels to min/max = -/+ 1
            packaged_labels = self._apply_label_normalization(packaged_labels, packaged_valid_t0_indices)

        if shuffle_indices:
            self.rng_generator.shuffle(packaged_valid_t0_indices)

        self.logger.info( "  Data tensors -> signals, labels, sample_indices, window_start_indices:")
        for tensor in [
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
        ]:
            self.logger.info(
                f"  shape {tensor.shape}, dtype {tensor.dtype}, min {tensor.min():.3f}, max {tensor.max():.3f}"
            )

        return_tuple = (
            packaged_signals, 
            packaged_labels, 
            packaged_valid_t0_indices, 
            packaged_window_start, 
            elm_indices,
        )
        # ensure everything is np.ndarray
        for item in return_tuple:
            assert isinstance(item, np.ndarray)

        return return_tuple

    # def _validate_data(self) -> None:

    #     # normalize signals to max ~= 1
    #     if self.normalize_signals:
    #         self.logger.info(f"  Normalizing signals to max ~= 1")
    #         packaged_signals /= 10.4  # normalize to max ~= 1
    #         packaged_signals /= 2*0.022  # set stdev to ~0.5

    #     self.signals_min = np.min(packaged_signals)
    #     self.signals_max = np.max(packaged_signals)


    def _apply_label_normalization(self) -> torch.Tensor:
        raise NotImplementedError

    def _check_for_balanced_data(self) -> None:
        # if classification, must implement in subclass
        raise NotImplementedError

    def _get_valid_indices(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_datasets(self) -> None:
        if hasattr(self, 'prediction_horizon'):
            prediction_horizon = self.prediction_horizon
        else:
            prediction_horizon = 0

        self.train_dataset = ELM_Dataset(
            signals=self.train_data[0],
            labels=self.train_data[1],
            sample_indices=self.train_data[2],
            signal_window_size = self.signal_window_size,
            prediction_horizon=prediction_horizon,
        )
        training_data_stats = self.train_dataset.get_signal_statistics()
        self.logger.info(
            f"Raw signals min {training_data_stats['min']:.4f} max {training_data_stats['max']:.4f} " +
            f"mean {training_data_stats['mean']:.4f} stdev {training_data_stats['stdev']:.4f}"
        )
        self.results['training_data_stats'] = training_data_stats
        if self.standardize_signals:
            self.logger.info("Standardizing training data with mean=0, stdev=1")
            self.train_dataset.standardize_signals(**training_data_stats)
            if self.clip_n_sigma:
                self.logger.info(f"  Clipping signal windows with abs(signals) > {self.clip_n_sigma} * stdev")
                new_stats = self.train_dataset.clip_signals(clip_n_sigma=self.clip_n_sigma)
                self.logger.info(f"  New min {new_stats['min']:.4f} max {new_stats['max']:.4f} mean {new_stats['mean']:.4f} stdev {new_stats['stdev']:.4f}")
                self.logger.info(f"  New sample indices size: {self.train_dataset.sample_indices.numel()}")

        if self.validation_data:
            self.validation_dataset = ELM_Dataset(
                signals=self.validation_data[0],
                labels=self.validation_data[1],
                sample_indices=self.validation_data[2],
                signal_window_size = self.signal_window_size,
                prediction_horizon=prediction_horizon,
            )
            if self.standardize_signals:
                self.logger.info("Standardizing validation data in accordance with training data")
                self.validation_dataset.standardize_signals(**training_data_stats)
                if self.clip_n_sigma:
                    self.logger.info(f"  Clipping signal windows with abs(signals) > {self.clip_n_sigma} * stdev")
                    new_stats = self.validation_dataset.clip_signals(clip_n_sigma=self.clip_n_sigma)
                    self.logger.info(f"  New min {new_stats['min']:.4f} max {new_stats['max']:.4f} mean {new_stats['mean']:.4f} stdev {new_stats['stdev']:.4f}")
                    self.logger.info(f"  New sample indices size: {self.validation_dataset.sample_indices.numel()}")
        else:
            self.validation_dataset = None

    def _make_data_loaders(self) -> None:
        train_sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            shuffle=True if self.seed is None else False,
            drop_last=True,
        ) if self.is_ddp else None
        self.train_data_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            sampler= train_sampler,
            batch_size=self.batch_size,
            shuffle=True if (self.seed is None and self.is_ddp is False) else False,
            num_workers=self.num_workers,
            # pin_memory=(self.device.type == 'cpu'),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True,
        )
        if self.validation_dataset:
            validation_sampler = torch.utils.data.DistributedSampler(
                self.validation_dataset,
                shuffle=False,
                drop_last=True,
            ) if self.is_ddp else None
            self.validation_data_loader = torch.utils.data.DataLoader(
                dataset=self.validation_dataset,
                sampler=validation_sampler,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                # pin_memory=(self.device.type == 'cpu'),
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=True,
            )


# TODO: make dataclass
class ELM_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        signals: np.ndarray = None, 
        labels: np.ndarray = None, 
        sample_indices: np.ndarray = None, 
        signal_window_size: int = None,
        prediction_horizon: int = 0,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = torch.unsqueeze(torch.from_numpy(signals), 0)
        assert (
            self.signals.ndim == 4 and 
            self.signals.shape[0] == 1 and 
            self.signals.shape[2] == 8 and 
            self.signals.shape[3] == 8
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        assert self.labels.ndim == 1, "Labels have incorrect shape"
        assert self.labels.shape[0] == self.signals.shape[1], "Labels and signals have different time dimensions"
        self.sample_indices = torch.from_numpy(sample_indices)
        self.signal_window_size = torch.tensor(signal_window_size, dtype=torch.int)
        self.prediction_horizon = torch.tensor(prediction_horizon, dtype=torch.int)

    def get_signal_statistics(self, skip_mean_stdev: bool = False) -> dict:
        signal_min = np.inf
        signal_max = -np.inf
        for signal_window, _ in self:
            signal_min = np.min([signal_min, signal_window.min()])
            signal_max = np.max([signal_max, signal_window.max()])
        if skip_mean_stdev:
            return_value = {
                'min': signal_min.item(),
                'max': signal_max.item(),
            }
        else:
            n_bins = 80
            cummulative_hist = np.zeros(n_bins, dtype=int)
            for signal_window, _ in self:
                hist, bin_edges = np.histogram(
                    signal_window,
                    bins=n_bins,
                    range=[signal_min, signal_max],
                )
                cummulative_hist += hist
            bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
            stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
            return_value = {
                'min': signal_min.item(),
                'max': signal_max.item(),
                'mean': mean.item(),
                'stdev': stdev.item(),
            }
        return return_value

    def standardize_signals(
        self,
        mean: float,
        stdev: float,
        **kwargs,
    ) -> None:
        self.signals = (self.signals - mean) / stdev

    def clip_signals(self, clip_n_sigma: float = 6.0) -> dict:
        sample_indices = self.sample_indices.tolist()
        for i in range(len(sample_indices)-1, -1, -1):
            signal_window, _ = self[i]
            if torch.max(torch.abs(signal_window)) > clip_n_sigma:
                sample_indices.pop(i)
        self.sample_indices = torch.tensor(sample_indices, dtype=torch.int)
        new_stats = self.get_signal_statistics()
        assert new_stats['min'] >= -clip_n_sigma and new_stats['max'] <= clip_n_sigma
        return new_stats

    def __len__(self) -> int:
        return self.sample_indices.size(dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time_idx = self.sample_indices[idx]
        # BES signal window data
        signal_window = self.signals[:, time_idx : time_idx + self.signal_window_size, :, :]
        # label for signal window
        label = self.labels[ time_idx + self.signal_window_size + self.prediction_horizon - 1 ]
        return signal_window, label

