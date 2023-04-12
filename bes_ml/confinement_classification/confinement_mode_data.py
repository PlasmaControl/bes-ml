from __future__ import annotations

from pathlib import Path
import dataclasses
import pickle

import numpy as np
import torch
import torch.utils.data
import h5py
import yaml
import matplotlib.pyplot as plt

import psutil
import os 

try:
    from .train_base import Trainer_Base_Dataclass
    from .models import Multi_Features_Model_Dataclass
    from .utilities import merge_pdfs
except ImportError:
    from bes_ml.base.train_base import Trainer_Base_Dataclass
    from bes_ml.base.models import Multi_Features_Model_Dataclass
    from bes_ml.base.utilities import merge_pdfs

@dataclasses.dataclass(eq=False)
class Confinement_Mode_Data(
    Trainer_Base_Dataclass,
    Multi_Features_Model_Dataclass,
):
    data_location: Path|str = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/small_confinement_data.hdf5'  # path to data; dir or file depending on task
    batch_size: int = 64  # power of 2, like 16-128
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    num_workers: int = None  # number of subprocess workers for pytorch dataloader
    pin_memory: bool = True  # data loader pinned memory
    seed: int = None  # RNG seed for deterministic, reproducible shuffling (ELMs, sample indices, etc.)
    label_type: np.int8 | np.float32 = dataclasses.field(default=None, init=False)
    test_data_file: str = 'test_data.pkl'
    standardize_signals: bool = True,  # if True, standardize signals based on training data mean~0, stdev~1
    standardize_fft: bool = True,  # if True, standardize FFTs based on training data log10(FFT^2) mean~0, stdev~1
    clip_sigma: float = 8.0  # remove signal windows with abs(standardized_signals) > n_sigma
    clip_signals: float = None # clip (remove) signals at +/- N Volts
    clamp_signals: float = None # clamp (keep windows) signals at +/- N Volts
    data_partition_file: str = 'data_partition.yaml'  # data partition for training, valid., and testing
    max_events: int = None

    def _prepare_data(self) -> None:

        self.data_location = Path(self.data_location).resolve()
        assert self.data_location.exists(), f"{self.data_location} does not exist"
        
        if self.is_ddp and self.seed is None:
            self.logger.info('Multi-GPU training requires identical shuffling; setting seed=0')
            self.seed = 0
        self.rng_generator = np.random.default_rng(seed=self.seed)

        if self.is_regression:
            # float labels for regression
            self.label_type = np.float32
        elif self.is_classification:
            # int labels for classification
            self.label_type = np.int8

        if self.num_workers is None:
            self.num_workers = 2 if self.device.type == 'cuda' else 0
        self.logger.info(f"Subprocess workers per data loader: {self.num_workers}")

        self._get_data()
        self._make_datasets()
        self._make_data_loaders()

    def _get_data(self) -> None:
        self._ddp_barrier()
        self.logger.info(f"Data file: {self.data_location}")

        with h5py.File(self.data_location, "r") as data_file:
            self.logger.info(f"confinement modes in data file: {len(data_file)}")
            good_keys = []
            for key in data_file:
                good_keys.append(key)
            indices = np.array(
                [int(key) for key in good_keys],
                dtype=int,
            )
            time_frames = sum([data_file[key]['signals'].shape[1] for key in good_keys])

        self.logger.info(f"confinement mode events: {indices.size}")
        self.logger.info(f"Total time frames for confinement mode events: {time_frames:,}")

        # shuffle confinement mode events
        self.rng_generator.shuffle(indices)

        if self.max_events:
            indices = indices[:self.max_events]
            self.logger.info(f"Limiting data to {self.max_events} confinement mode events")

        if indices.size >= 5:
            self.logger.info(f"Initial confinement mode indices: {indices[0:5]}")

        n_validation_confinement_modes = int(self.fraction_validation * indices.size)
        n_test_confinement_modes = int(self.fraction_test * indices.size)

        test_confinement_modes, validation_confinement_modes, training_confinement_modes = np.split(
            indices,
            [n_test_confinement_modes, n_test_confinement_modes+n_validation_confinement_modes]
        )

        with (self.output_dir/self.data_partition_file).open('w') as data_partition_file:
            data_partition = {
                'n_confinement_modes': indices.size,
                'data_location': self.data_location.as_posix(),
                'training_confinement_modes': training_confinement_modes.tolist(),
                'validation_confinement_modes': validation_confinement_modes.tolist(),
                'test_confinement_modes': test_confinement_modes.tolist(),
            }
            yaml.safe_dump(
                data_partition,
                data_partition_file,
                default_flow_style=False,
                sort_keys=False,
            )

        self._ddp_barrier()
        self.logger.info(f"Training data confinement mode events: {training_confinement_modes.size}")

        self.train_data = self._preprocess_data(
            indices=training_confinement_modes,
            shuffle_indices=True,
            # oversample_active_elm=self.oversample_active_elm if self.is_classification else False,
            is_train_data=True,
        )

        if n_validation_confinement_modes:
            self._ddp_barrier()
            self.logger.info(f"Validation data confinement mode events: {validation_confinement_modes.size}")
            self.validation_data = self._preprocess_data(
                indices=validation_confinement_modes,
                # save_filename='validation_confinement_modes',
            )
        else:
            self.logger.info("Skipping validation data")
            self.validation_data = None

        if n_test_confinement_modes and self.is_main_process:
            self.logger.info(f"Test data confinement mode events: {test_confinement_modes.size}")
            self.test_data = self._preprocess_data(
                indices=test_confinement_modes,
                # save_filename='test_confinement_modes',
                is_test_data=True,
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
                        "indices": self.test_data[4],
                    },
                    file,
                )
            self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")
        else:
            self.logger.info("Skipping test data")
            self.test_data = None

    def _preprocess_data(
        self,
        indices: np.ndarray = None,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
        save_filename: str = '',
        is_train_data: bool = False,
        is_test_data: bool = False,
    ) -> tuple:
        if save_filename and self.is_main_process:
            plt.ioff()
            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
            self.logger.info(f"  Plotting valid indices: {save_filename}_**.pdf")
            i_page = 1
            axes_twinx = [axis.twinx() for axis in axes.flat]
        with h5py.File(self.data_location, 'r') as h5_file:
            confinement_mode_data = []
            time_counts = []
            for i_confinement_mode, confinement_mode_index in enumerate(indices):
                confinement_mode_key = f"{confinement_mode_index:05d}"
                time_counts.append(h5_file[confinement_mode_key]["signals"].shape[1])
            time_count = np.sum(time_counts)
            packaged_signals = np.empty((time_count, 6, 8), dtype=np.float32)
            start_index = 0
            for i_confinement_mode, confinement_mode_index in enumerate(indices):
                if i_confinement_mode%10 == 0:
                    self.logger.info(f"  confinement mode event {i_confinement_mode:04d}/{indices.size:04d}")
                confinement_mode_key = f"{confinement_mode_index:05d}"
                confinement_mode_event = h5_file[confinement_mode_key]                
                signals = np.array(confinement_mode_event["signals"], dtype=np.float32)  # (48, <time>)
                signals = np.transpose(signals, (1, 0)).reshape(-1, 6, 8)  # reshape to (<time>, 6, 8)
                labels = np.array(confinement_mode_event["labels"], dtype=self.label_type)
                labels, valid_t0 = self._get_valid_indices(labels)
      
                # clamp (keep) signals at +/- 2.0 V
                if self.clamp_signals:
                    signals[np.where(signals>self.clamp_signals)] = self.clamp_signals
                    signals[np.where(signals<-self.clamp_signals)] = -self.clamp_signals

                packaged_signals[start_index:start_index + signals.shape[0]] = signals
                start_index += signals.shape[0]
                if save_filename and self.is_main_process:
                    if i_confinement_mode % 12 == 0:
                        for i_axis in range(axes.size):
                            axes.flat[i_axis].clear()
                            axes_twinx[i_axis].clear()
                    twinx = axes_twinx[i_confinement_mode%12]
                    twinx.plot(signals[:,2,3]/10, label='BES 20', color='C1', zorder=0)
                    twinx.plot(signals[:,2,5]/10, label='BES 22', color='C2', zorder=0)
                    twinx.set_ylabel('Raw signal/10')
                    twinx.legend(fontsize='x-small', loc='upper right')
                    plt.sca(axes.flat[i_confinement_mode%12])
                    # plt.plot(labels, label='Label', color='C0')
                    if 0 in labels:
                        mode = 'L-mode'
                    if 1 in labels:
                        mode = 'H-mode'
                    if 2 in labels:
                        mode = 'QH-mode'
                    if 3 in labels:
                        mode = 'WP QH-mode'
                    if self.is_classification:
                        plt.ylabel(mode)
                    else:
                        plt.ylabel('Time to ELM onset (mu-s)')
                    plt.legend(fontsize='x-small', loc='upper left')
                    # plt.title(f"Confinement mode index {confinement_mode_key}")
                    plt.title("Shot "+str(confinement_mode_key)[:6]+" at time "+str(confinement_mode_key)[6:])
                    plt.xlabel('Time (mu-s)')
                    if i_confinement_mode%12==11 or i_confinement_mode==indices.size-1:
                        plt.tight_layout()
                        output_file = self.output_dir/(save_filename + f"_{i_page:02d}.pdf")
                        plt.savefig(
                            output_file, 
                            format="pdf", 
                            transparent=True,
                        )
                        i_page += 1
                confinement_mode_data.append(
                    {'labels': labels, 'valid_t0': valid_t0}
                )

        self.logger.info('  Finished reading confinement mode event data')
        
        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in confinement_mode_data], axis=0)
        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in confinement_mode_data], axis=0)
        index_count = 0
        packaged_window_start = np.array([], dtype=int)
        for confinement_mode in confinement_mode_data:
            packaged_window_start = np.append(
                packaged_window_start,
                index_count,
            )
            index_count += confinement_mode['labels'].size

        assert packaged_labels.size == packaged_valid_t0.size

        if save_filename and self.is_main_process:
            plt.close()
            pdf_files = sorted(self.output_dir.glob(f'{save_filename}_*.pdf'))
            output = self.output_dir / f'{save_filename}.pdf'
            merge_pdfs(pdf_files, output, delete_inputs=True)
        
        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices + self.signal_window_size]))

        # get signal stats
        stats = self._get_statistics(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )
        self.logger.info(f" Raw signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        # clip at +/- N volts
        if self.clip_signals:
            self.logger.info(f"  -> Clipping signal windows beyond +/- {self.clip_signals} V")
            mask = []
            for i in packaged_valid_t0_indices:
                signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                mask.append((signal_window.min() >= -self.clip_signals) and (signal_window.max() <= self.clip_signals))
            packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            self.logger.info(f"  Clipped signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        if self.mlp_output_size == 3:
            if not is_test_data:
                for local_gpu in range(torch.cuda.device_count()):
                    if self.local_rank == local_gpu:
                        packaged_labels = packaged_labels - 1
                    self._ddp_barrier()
            else:
                packaged_labels = packaged_labels - 1

        if self.mlp_output_size == 1:
            packaged_labels = packaged_labels - packaged_labels.min()
            packaged_labels[np.where(packaged_labels>0)[0]] = packaged_labels[np.where(packaged_labels>0)[0]]/packaged_labels.max()

        if is_train_data:
            self.results['raw_train_signal_mean'] = stats['mean']
            self.results['raw_train_signal_stdev'] = stats['stdev']
        del confinement_mode_data

        # standardize signals with mean~0 and stdev~1
        if self.standardize_signals:
            self.logger.info(f" Signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")
            assert self.results['raw_train_signal_mean'] and self.results['raw_train_signal_stdev']
            mean = self.results['raw_train_signal_mean']
            stdev = self.results['raw_train_signal_stdev']

            self.logger.info(f"  -> Standardizing signals with mean {mean:.4f} and stdev {stdev:.4f} from training data")
            
            if not is_test_data:
                for local_gpu in range(torch.cuda.device_count()):
                    if self.local_rank == local_gpu:
                        packaged_signals = (packaged_signals - mean) / stdev
                    self._ddp_barrier()
            else:
                packaged_signals = (packaged_signals - mean) / stdev
                
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
            self.logger.info(f"  Standardized signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")
            # clip at +/- sigma
            if self.clip_sigma:
                self.logger.info(f"  -> Clipping signal windows beyond +/- {self.clip_sigma} sigma")
                mask = []
                for i in packaged_valid_t0_indices:
                    signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                    mask.append((signal_window.min() >= -self.clip_sigma) and (signal_window.max() <= self.clip_sigma))
                packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
                stats = self._get_statistics(
                    sample_indices=packaged_valid_t0_indices,
                    signals=packaged_signals,
                )
                self.logger.info(f"  Clipped signals count {stats['count']} min {stats['min']:.4f} max {stats['max']:.4f} mean {stats['mean']:.4f} stdev {stats['stdev']:.4f}")

        if self.fft_num_kernels:
            self.model.fft_features.fft_calc_histogram = True
            self.model.fft_features.reset_histogram()
            stat_interval = packaged_valid_t0_indices.size // 1000
            if stat_interval < 1:
                stat_interval = 1
            for i in packaged_valid_t0_indices[::stat_interval]:
                signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                signal_window = torch.from_numpy(signal_window[np.newaxis, np.newaxis, ...]).to(self.device)
                _ = self.model.fft_features.forward(signal_window)
            bin_edges = self.model.fft_features.bin_edges
            cummulative_hist = self.model.fft_features.cummulative_hist
            bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
            stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
            self.logger.info(f"  log10(|FFT|^2)  mean {mean:.4f}  stdev {stdev:.4f}")
            if is_train_data and self.standardize_fft:
                self.logger.info(f"  -> Standardizing FFT in data with train data FFTs")
                self.results['train_signal_fft_mean'] = mean.item()
                self.results['train_signal_fft_stdev'] = stdev.item()
                assert (
                    self.model.fft_features.fft_mean is None and
                    self.model.fft_features.fft_stdev is None
                )
                self.model.fft_features.fft_mean = self.results['train_signal_fft_mean']
                self.model.fft_features.fft_stdev = self.results['train_signal_fft_stdev']
                # redo hist calculation with mean/stdev for standardization
                self.model.fft_features.reset_histogram()
                for i in packaged_valid_t0_indices[::stat_interval]:
                    signal_window = packaged_signals[i: i + self.signal_window_size, :, :]
                    signal_window = torch.from_numpy(signal_window[np.newaxis, np.newaxis, ...]).to(self.device)
                    _ = self.model.fft_features.forward(signal_window)
                bin_edges = self.model.fft_features.bin_edges
                cummulative_hist = self.model.fft_features.cummulative_hist
                bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
                mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
                stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
                self.logger.info(f"  Standardized log10(|FFT|^2)  mean {mean:.4f}  stdev {stdev:.4f}")
            self.model.fft_features.fft_calc_histogram = False

        # balance or normalize labels
        if self.is_classification:
            pass
        #     # assess data balance for active ELM/inactive ELM classification
        #     packaged_valid_t0_indices = self._check_for_balanced_data(
        #         packaged_labels=packaged_labels,
        #         packaged_valid_t0_indices=packaged_valid_t0_indices,
        #         oversample_active_elm=oversample_active_elm,
            # )
        elif self.is_regression:
            # if specified, normalize time-to-ELM labels to min/max = -/+ 1
            # packaged_labels = self._apply_label_normalization(
            #     packaged_labels,
            #     packaged_valid_t0_indices,
            # )
            raw_label_min = packaged_labels[packaged_valid_t0_indices+self.signal_window_size].min()
            raw_label_max = packaged_labels[packaged_valid_t0_indices+self.signal_window_size].max()
            raw_label_median = np.median(packaged_labels[packaged_valid_t0_indices+self.signal_window_size])
            self.logger.info(f"  Raw label min/max: {raw_label_min:.4e}, {raw_label_max:.4e}")
            self.logger.info(f"  Raw label median: {raw_label_median:.4e}")
            if is_train_data:
                self.results['raw_label_min'] = raw_label_min.item()
                self.results['raw_label_max'] = raw_label_max.item()
                self.results['raw_label_median'] = raw_label_median.item()
            if self.normalize_labels:
                # self.logger.info(f"  -> Normalizing labels to min/max = -/+ 1 based on training data")
                self.logger.info(f"  -> Normalizing labels to min=-1 and median=0 based on training data")
                packaged_labels = packaged_labels - self.results['raw_label_median']
                packaged_labels = packaged_labels / (self.results['raw_label_median'] - self.results['raw_label_min'])
                raw_label_min = packaged_labels[packaged_valid_t0_indices+self.signal_window_size].min()
                raw_label_max = packaged_labels[packaged_valid_t0_indices+self.signal_window_size].max()
                raw_label_median = np.median(packaged_labels[packaged_valid_t0_indices+self.signal_window_size])
                self.logger.info(f"  Norm. label min/max: {raw_label_min:.4e}, {raw_label_max:.4e}")
                self.logger.info(f"  Norm. label median: {raw_label_median:.4e}")
                # label_range = self.results['raw_label_max'] - self.results['raw_label_min']
                # packaged_labels = ((packaged_labels - self.results['raw_label_min']) / label_range - 0.5) * 2
                assert np.all(packaged_labels[packaged_valid_t0_indices+self.signal_window_size]>=-1)
                assert np.min(packaged_labels[packaged_valid_t0_indices+self.signal_window_size]) == -1
                # if is_train_data:
                #     assert np.all(packaged_labels[packaged_valid_t0_indices+self.signal_window_size]<=1)
                #     assert np.max(packaged_labels[packaged_valid_t0_indices+self.signal_window_size]) == 1

        if shuffle_indices:
            self.rng_generator.shuffle(packaged_valid_t0_indices)

        self.logger.info("  Finished packaging data")

        self.logger.info( "  Data tensors -> signals, labels, sample_indices, window_start_indices, indices:")
        return_tuple = (
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
            indices,
        )
        for array in return_tuple:
            assert isinstance(array, np.ndarray)
            self.logger.info(
                f"    shape {array.shape}, dtype {array.dtype}, min {np.nanmin(array):.3f}, max {np.nanmax(array):.3f}"
            )

        return return_tuple

    def _get_statistics(self, sample_indices: np.ndarray, signals: np.ndarray) -> dict:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_interval = sample_indices.size // 1000 if sample_indices.size > 10000 else 1
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
        return {
            'count': sample_indices.size,
            'min': signal_min.item(),
            'max': signal_max.item(),
            'mean': mean.item(),
            'stdev': stdev.item(),
        }

    # def _apply_label_normalization(self) -> torch.Tensor:
    #     raise NotImplementedError

    def _check_for_balanced_data(self) -> None:
        # if classification, must implement in subclass
        raise NotImplementedError

    def _get_valid_indices(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_datasets(self) -> None:
        self._ddp_barrier()
        self.logger.info('Making datasets')
        self.train_dataset = Confinement_Mode_Dataset(
            signals=self.train_data[0],
            labels=self.train_data[1],
            sample_indices=self.train_data[2],
            signal_window_size = self.signal_window_size,
            prediction_horizon=self.prediction_horizon if hasattr(self, 'prediction_horizon') else 0,
        )

        self._ddp_barrier()
        self.validation_dataset = Confinement_Mode_Dataset(
            signals=self.validation_data[0],
            labels=self.validation_data[1],
            sample_indices=self.validation_data[2],
            signal_window_size = self.signal_window_size,
            prediction_horizon=self.prediction_horizon if hasattr(self, 'prediction_horizon') else 0,
        ) if self.validation_data else None

    def _make_data_loaders(self) -> None:
        self._ddp_barrier()
        self.train_sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            shuffle=(self.seed is None),
            drop_last=True,
        ) if self.is_ddp else None
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=(self.seed is None and self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )
        if self.validation_dataset:
            self._ddp_barrier()
            self.valid_sampler = torch.utils.data.DistributedSampler(
                self.validation_dataset,
                shuffle=False,
                drop_last=True,
            ) if self.is_ddp else None
            self.valid_loader = torch.utils.data.DataLoader(
                dataset=self.validation_dataset,
                sampler=self.valid_sampler,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )


class Confinement_Mode_Dataset(torch.utils.data.Dataset):

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
            self.signals.size(2) == 6 and
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

    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.sample_indices[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        label = self.labels[ i_t0 + self.signal_window_size + self.prediction_horizon - 1 ]
        return signal_window, label
