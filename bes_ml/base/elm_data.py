import logging
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
    from .models import _Base_Features_Dataclass, _Multi_Features_Model_Dataclass
except ImportError:
    from bes_ml.base.train_base import _Base_Trainer_Dataclass
    from bes_ml.base.utilities import merge_pdfs
    from bes_ml.base.models import _Base_Features_Dataclass, _Multi_Features_Model_Dataclass


@dataclasses.dataclass(eq=False)
class _ELM_Data_Base(
    _Base_Trainer_Dataclass,
    _Multi_Features_Model_Dataclass,
):
    data_location: Union[Path,str] = sample_elm_data_file  # path to data; dir or file depending on task
    batch_size: int = 64  # power of 2, like 16-128
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    test_data_file: str = 'test_data.pkl'
    standardize_signals: bool = True,
    standardize_fft: bool = True,
    clip_sigma: float = 8.0  # remove signal windows with abs(standardized_signals) > n_sigma
    seed: int = None  # RNG seed for deterministic, reproducible shuffling (ELMs, sample indices, etc.)
    data_partition_file: str = 'data_partition.yaml'  # data partition for training, valid., and testing
    max_elms: int = None
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    pin_memory: bool = True  # data loader pinned memory
    bad_elm_indices: Iterable = None  # iterable of ELM indices to skip when reading data
    bad_elm_indices_csv: str | bool = True  # CSV file to read bad ELM indices
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
        self._make_datasets()
        self._make_data_loaders()

    def _get_data(self) -> None:
        self.logger.info(f"Data file: {self.data_location}")

        with h5py.File(self.data_location, "r") as data_file:
            self.logger.info(f"ELM events in data file: {len(data_file)}")
            if self.bad_elm_indices is None:
                self.bad_elm_indices = []
            if self.bad_elm_indices_csv is True:
                self.bad_elm_indices_csv = bad_elm_indices_csv
            if self.bad_elm_indices_csv:
                self.logger.info(f"Ignoring bad ELM indices from {self.bad_elm_indices_csv}")
                with Path(self.bad_elm_indices_csv).open() as file:
                    self.bad_elm_indices = [int(line) for line in file]
            good_keys = []
            bad_elm_count = 0
            for key in data_file:
                if int(key) not in self.bad_elm_indices:
                    good_keys.append(key)
                else:
                    bad_elm_count += 1
            elm_indices = np.array(
                [int(key) for key in good_keys],
                dtype=int,
            )
            time_frames = sum([data_file[key]['time'].shape[0] for key in good_keys])
        if bad_elm_count:
            self.logger.info(f"Ignored bad ELM events: {bad_elm_count}")
        self.logger.info(f"Valid ELM events: {elm_indices.size}")
        self.logger.info(f"Total time frames for valid ELM events: {time_frames:,}")

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

        self.logger.info(f"Training ELM events: {training_elms.size}")
        self.train_data = self._preprocess_data(
            elm_indices=training_elms,
            shuffle_indices=True,
            oversample_active_elm=self.oversample_active_elm if self.is_classification else False,
        )
        self._barrier()

        if n_validation_elms:
            self.logger.info(f"Validation ELM events: {validation_elms.size}")
            self.validation_data = self._preprocess_data(
                elm_indices=validation_elms,
                save_filename='validation_elms',
            )
        else:
            self.logger.info("Skipping validation data")
            self.validation_data = None
        self._barrier()

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
            self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")
        else:
            self.logger.info("Skipping test data")
            self.test_data = None
        self._barrier()

    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
        save_filename: str = '',
    ) -> tuple:
        if save_filename and self.is_main_process:
            plt.ioff()
            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
            self.logger.info(f"  Plotting valid indices: {save_filename}_**.pdf")
            i_page = 1
        with h5py.File(self.data_location, 'r') as h5_file:
            elm_data = []
            for i_elm, elm_index in enumerate(elm_indices):
                if i_elm%200 == 0:
                    self.logger.info(f"  ELM event {i_elm:04d}/{elm_indices.size:04d}")
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
                elm_data.append(
                    {'signals': signals, 'labels': labels, 'valid_t0': valid_t0}
                )
                if save_filename and self.is_main_process:
                    if i_elm % 12 == 0:
                        for axis in axes.flat:
                            plt.sca(axis)
                            plt.cla()
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

        self.logger.info('  Finished reading ELM event data')

        packaged_labels = np.concatenate([elm['labels'] for elm in elm_data], axis=0)
        packaged_signals = np.concatenate([elm['signals'] for elm in elm_data], axis=0)
        packaged_valid_t0 = np.concatenate([elm['valid_t0'] for elm in elm_data], axis=0)
        index_count = 0
        packaged_window_start = np.array([], dtype=int)
        for elm in elm_data:
            packaged_window_start = np.append(
                packaged_window_start,
                index_count,
            )
            index_count += elm['labels'].size

        assert packaged_labels.size == packaged_valid_t0.size

        if save_filename and self.is_main_process:
            plt.close()
            pdf_files = sorted(self.output_dir.glob(f'{save_filename}_*.pdf'))
            output = self.output_dir / f'{save_filename}.pdf'
            merge_pdfs(pdf_files, output, delete_inputs=True)
        
        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        # standardize/clip signals
        stats = self._get_statistics(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )
        self.logger.info(f"  Raw signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        if self.clip_sigma:
            self.logger.info(f"  Clipping signal windows beyond +/- {self.clip_sigma} sigma")
            mask = []
            lb = stats['mean'] - self.clip_sigma * stats['stdev']
            ub = stats['mean'] + self.clip_sigma * stats['stdev']
            for i in packaged_valid_t0_indices:
                signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                mask.append((signal_window.min() >= lb) and (signal_window.max() <= ub))
            packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            self.logger.info(f"  Clipped signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        self.results['raw_signal_mean'] = stats['mean']
        self.results['raw_signal_stdev'] = stats['stdev']

        if self.standardize_signals:
            self.logger.info(f"  Standardizing signals with mean {stats['mean']:.4f} and stdev {stats['stdev']:.4f}")
            packaged_signals = (packaged_signals - stats['mean']) / stats['stdev']
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            self.logger.info(f"  Standardized signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        if self.standardize_fft and self.fft_num_kernels:
            nfft = self.signal_window_size // self.fft_nbins
            nfreqs = nfft // 2 + 1
            hist_bins = 230
            cummulative_hist = np.zeros(hist_bins, dtype=int)
            fft_bins = np.empty((self.fft_nbins, nfreqs-1, 8, 8), dtype=np.float32)
            for i in packaged_valid_t0_indices:
                signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                for i_bin in range(self.fft_nbins):
                    rfft = np.fft.rfft(
                        signal_window[i_bin * nfft:(i_bin+1) * nfft, :, :], 
                        axis=0,
                    )
                    fft_bins[i_bin: i_bin + 1, :, :, :] = np.abs(rfft[1:, :, :]) ** 2
                fft_sw = np.mean(fft_bins, axis=0)
                fft_sw = np.log10(fft_sw)
                hist, bin_edges = np.histogram(
                    fft_sw,
                    bins=hist_bins,
                    range=[-7,7],
                )
                cummulative_hist += hist
            bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
            stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
            self.logger.info(f"  Original log10(|FFT|^2) mean {mean:.4f}  stdev {stdev:.4f}")
            self.logger.info(f"  Standardizing FFT in model")
            self.model.fft_features.fft_mean = mean
            self.model.fft_features.fft_stdev = stdev


        # balance or normalize labels
        if self.is_classification:
            # assess data balance for active ELM/inactive ELM classification
            packaged_valid_t0_indices = self._check_for_balanced_data(
                packaged_labels=packaged_labels,
                packaged_valid_t0_indices=packaged_valid_t0_indices,
                oversample_active_elm=oversample_active_elm,
            )
        elif self.is_regression:
            # if specified, normalize time-to-ELM labels to min/max = -/+ 1
            packaged_labels = self._apply_label_normalization(
                packaged_labels,
                packaged_valid_t0_indices,
            )

        if shuffle_indices:
            self.rng_generator.shuffle(packaged_valid_t0_indices)

        self.logger.info("  Finished packaging data")

        self.logger.info( "  Data tensors -> signals, labels, sample_indices, window_start_indices, elm_indices:")
        return_tuple = (
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
            elm_indices,
        )
        for array in return_tuple:
            assert isinstance(array, np.ndarray)
            self.logger.info(
                f"    shape {array.shape}, dtype {array.dtype}, min {array.min():.3f}, max {array.max():.3f}"
            )

        return return_tuple

    def _get_statistics(self, sample_indices: np.ndarray, signals: np.ndarray) -> dict:
        signal_min = np.inf
        signal_max = -np.inf
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        for i in sample_indices[::32]:
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
        return {
            'count': sample_indices.size,
            'min': signal_min.item(),
            'max': signal_max.item(),
            'mean': mean.item(),
            'stdev': stdev.item(),
        }

    def _apply_label_normalization(self) -> torch.Tensor:
        raise NotImplementedError

    def _check_for_balanced_data(self) -> None:
        # if classification, must implement in subclass
        raise NotImplementedError

    def _get_valid_indices(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_datasets(self) -> None:
        self.logger.info('Making datasets')
        self.train_dataset = ELM_Dataset(
            signals=self.train_data[0],
            labels=self.train_data[1],
            sample_indices=self.train_data[2],
            signal_window_size = self.signal_window_size,
            prediction_horizon=self.prediction_horizon if hasattr(self, 'prediction_horizon') else 0,
        )
        self._barrier()

        self.validation_dataset = ELM_Dataset(
            signals=self.validation_data[0],
            labels=self.validation_data[1],
            sample_indices=self.validation_data[2],
            signal_window_size = self.signal_window_size,
            prediction_horizon=self.prediction_horizon if hasattr(self, 'prediction_horizon') else 0,
        ) if self.validation_data else None
        self._barrier()

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
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
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
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )
        self._barrier()


# TODO: make dataclass
class ELM_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            labels: np.ndarray,
            sample_indices: np.ndarray,
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
        self.sample_indices = torch.from_numpy(sample_indices)
        assert torch.max(self.sample_indices)+self.signal_window_size+self.prediction_horizon-1 <= self.labels.numel()

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, i: int) -> Tuple:
        i_t0 = self.sample_indices[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        label = self.labels[ i_t0 + self.signal_window_size + self.prediction_horizon - 1 ]
        return signal_window, label

