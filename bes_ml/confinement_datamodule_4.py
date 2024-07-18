from __future__ import annotations
import dataclasses
from pathlib import Path
import gc
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import firwin, filtfilt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from scipy.stats import mode

import h5py

import torch
import torch.nn
import torch.utils.data
import time

from lightning.pytorch import LightningDataModule

from bes_data.sample_data import sample_elm_data_file

import psutil


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
    

class Confinement_Predict_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            labels: np.ndarray,
            signal_window_size: int,
            shot: int,
            start_time: int,
            confinement_mode_index: int,
    ) -> None:
        self.shot = shot
        self.start_time = start_time
        self.confinement_mode_index = confinement_mode_index
        self.signals = torch.from_numpy(signals[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == 2 and
            self.signals.size(3) == 8
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        assert self.labels.ndim == 1, "Labels have incorrect shape"
        assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
        self.signal_window_size = signal_window_size
        last_signal_window_start_index = self.labels.numel() - self.signal_window_size
        assert last_signal_window_start_index+self.signal_window_size == self.labels.numel()
        valid_t0 = np.zeros(self.labels.numel(), dtype=int)  
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start 
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start 
        sample_indices = np.arange(valid_t0.size, dtype=int)
        sample_indices = sample_indices[valid_t0 == 1]
        self.sample_indices = torch.from_numpy(sample_indices)

    def pre_elm_stats(self) -> dict[str, torch.Tensor]:
        pre_elm_signals = self.signals[0,:self.active_elm_start_index,...]
        maxabs, _ = torch.max(torch.abs(pre_elm_signals), dim=0)
        std, mean = torch.std_mean(pre_elm_signals, dim=0)
        return {
            'maxabs': maxabs.numpy(force=True),
            'mean': mean.numpy(force=True),
            'std': std.numpy(force=True),
        }

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.sample_indices[i]
        signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
        label_index = i_t0 + self.signal_window_size - 1
        label = self.labels[ label_index : label_index + 1 ]
        # label_class = torch.tensor([0]) if label >= 0 else torch.tensor([1])
        return signal_window, label, self.shot, self.start_time


@dataclasses.dataclass(eq=False)
class Confinement_Datamodule(LightningDataModule):
    data_file: str = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/6x8_confinement_data_8.hdf5'  # path to data; dir or file depending on task
    n_rows: int = 6
    n_cols: int = 8
    batch_size: int = 128  # power of 2, like 32-256
    signal_window_size: int = 128  # power of 2, like 64-512
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    test_only: bool = False
    seed: int = 0  # RNG seed for deterministic, reproducible shuffling of events
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }    
    max_shots: int = None
    max_shots_per_class: int = None
    num_classes: int = 4
    max_predict_confinement_modes: int = 24
    bad_shots: list = None
    force_validation_shots: list = None
    force_test_shots: list = None
    # Bandpass filter parameters
    sampling_frequency_hz: float = 1 / 10**(-6)  # Sampling frequency in Hz
    filter_taps: int = 501  # Number of taps in the filter
    lower_cutoff_frequency_hz: float = None  # Lower cutoff frequency in Hz
    upper_cutoff_frequency_hz: float = None  # Upper cutoff frequency in Hz
    clip_signals: float = None # remove signal windows with abs(raw_signals) > clip_signals
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    # bad_confinement_indices: list = None  # iterable of indices to skip when reading data
    # bad_confinement_indices_csv: str | bool = False  # CSV file to read bad indices
    # log_time: bool = False  # if True, use label = log(time_to_elm_onset)
    one_hot_labels: bool = False # if True, use one-hot vector for label
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    plot_data_stats: bool = True
    is_global_zero: bool = dataclasses.field(default=True, init=False)
    log_dir: str = dataclasses.field(default='.', init=False)
    world_size: int = 1 # number of total GPUs 

    def __post_init__(self):
        super().__init__()
        if self.data_file is None:
            self.data_file = sample_elm_data_file.as_posix()
        self.save_hyperparameters(
            ignore=['max_predict_elms']
        )

        # datamodule state, to reproduce pre-processing
        self.state_items = [
            'mask_ub',
            'mask_lb',
            'signal_mean',
            'signal_stdev',
            'signal_exkurt',
            'label_median',
        ]
        for item in self.state_items:
            if not hasattr(self, item):
                setattr(self, item, None)

        self.datasets = {}
        self.all_confinement_events = None
        self.test_confinement_events = None
        self.train_confinement_events = None
        self.validation_confinement_events = None
        self._get_confinement_events_and_split()
        self.dataset_confinement_events = {
                'train': self.train_confinement_events,
                'validation': self.validation_confinement_events,
                'test': self.test_confinement_events,
                'predict': self.test_confinement_events,
            }

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

    def state_dict(self) -> dict:
        state = {}
        for item in self.state_items:
            state[item] = getattr(self, item)
        return state

    def load_state_dict(self, state: dict) -> None:
        print("Loading state_dict")
        for item in self.state_items:
            setattr(self, item, state[item])
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"Running Confinement_Datamodule.setup(stage={stage})")
        # Determine the rank of this GPU
        try:
            local_rank = self.trainer.local_rank
            node_rank = self.trainer.node_rank
        except:
            local_rank = int(os.getenv('SLURM_LOCALID', 0))
            node_rank = int(os.getenv('SLURM_NODEID', 0))

        global_rank = node_rank * 4 + local_rank

        # Determine the dataset stage (train, validation, or test) based on the current stage
        if stage == 'fit':
            dataset_stages = ['train', 'validation']
        elif stage == 'test' or stage == 'predict':
            dataset_stages = [stage]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Create DataLoaders for each dataset stage
        for dataset_stage in dataset_stages:
            # Determine the chunk of confinement indices for this GPU
            events = self.dataset_confinement_events[dataset_stage]
            times = [self.get_time_for_index(shot_event) for shot_event in events]  # Adapted for (shot, event) tuples
            if dataset_stage in ['train']:
                print(f"Creating chunks for {dataset_stage} with {len(events)} indices and total time {sum(times)}")
                # Create balanced chunks
                chunks = self.create_balanced_chunks(events, times, self.world_size)
                # Determine the chunk for this GPU
                chunk_events = chunks[global_rank]
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
        
    def get_time_for_index(self, shot_event_tuple):
        shot, event = shot_event_tuple  # Unpack the tuple
        with h5py.File(self.data_file, 'r') as h5_file:
            event_key = f"{shot}/{event}"  # Updated to use shot/event structure
            time_count = h5_file[event_key]["signals"].shape[1]
        return time_count

    def create_balanced_chunks(self, indices, times, num_chunks):
        # Create a mapping from indices to times
        index_to_time = {index: time for index, time in zip(indices, times)}

        # Create a list to hold the chunks, and a list to hold the total time for each chunk
        chunks = [[] for _ in range(num_chunks)]
        chunk_times = [0] * num_chunks

        # Iterate over the indices, sorted by time from largest to smallest
        for index, time in sorted(index_to_time.items(), key=lambda item: item[1], reverse=True):
            # Find the chunk with the shortest total time so far
            min_time_chunk_idx = min(range(num_chunks), key=lambda i: chunk_times[i])

            # Add this index to that chunk
            chunks[min_time_chunk_idx].append(index)

            # Update the total time for that chunk
            chunk_times[min_time_chunk_idx] += time

        # Print information about the chunks
        for i, (chunk, chunk_time) in enumerate(zip(chunks, chunk_times)):
            print(f"Chunk {i} size: {len(chunk)}, total time: {sum(index_to_time[index] for index in chunk)}")

        return chunks
        
    def _load_and_preprocess_data_2(self, shot_event_indices, dataset_stage):
        t0 = time.time()
        print(f"Reading confinement events for dataset `{dataset_stage}`")
        confinement_data = []
        n_bins = 201
        cummulative_hist = np.zeros(n_bins, dtype=int)
        selected_channels = self.n_rows * self.n_cols  # Total number of channels to select

        with h5py.File(self.data_file, 'r') as h5_file:
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
            for i, (shot, event) in enumerate(long_enough_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(shot_event_indices):04d} in shot {shot}")
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Retrieve signals and reshape according to inboard_order
                signals = np.array(event_data["signals"][:, :], dtype=np.float32)
                # signals = self.reshape_signals_6x8(signals, inboard_order)
                # start_col_index = (8-self.n_cols)
                # signals = signals[:, :self.n_rows, start_col_index:]
                
                signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = self.apply_bandpass_filter(signals)
                labels = np.array(event_data["labels"], dtype=int)

                # maxabs_by_channel = np.amax(np.abs(signals[:,:,:]), axis=0)
                # min_max_mask = (
                #     np.isclose(signals[:,:4,:], 10.375800) |
                #     np.isclose(signals[:,:4,:], -10.376433) 
                #     # np.isclose(signals[:,4:,:], 5.186306) |
                #     # np.isclose(signals[:,4:,:], -5.405264)
                # )
                # maxcount_by_channel = np.count_nonzero(min_max_mask, axis=0)
                # mean_by_channel = np.mean(signals[:,:,:], axis=0)
                # std_by_channel = np.std(signals[:,:,:], axis=0)
                # kurt_by_channel = scipy.stats.kurtosis(signals[:,:,:], axis=0, fisher=False)
                labels, valid_t0 = self._get_valid_indices(labels)
                packaged_signals[start_index:start_index + signals.shape[0]] = signals
                start_index += signals.shape[0]
                confinement_data.append({
                    'labels': labels, 
                    'valid_t0': valid_t0,
                    'confinement_mode_key': event_key,
                    'shot': shot,
                    'time': event,
                    # 'maxabs_by_channel': maxabs_by_channel,
                    # 'maxcount_by_channel': maxcount_by_channel,
                    # 'mean_by_channel': mean_by_channel,
                    # 'std_by_channel': std_by_channel,
                    # 'kurt_by_channel': kurt_by_channel,
                })
                # hist, bin_edges = np.histogram(
                #     signals[::50, :, :],
                #     bins=n_bins,
                #     range=[-10.4, 10.4],
                # )
                # cummulative_hist += hist

        # bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        # mean_all_data = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
        # stdev_all_data = np.sqrt(np.sum(cummulative_hist * (bin_center - mean_all_data) ** 2) / np.sum(cummulative_hist))
        # exkurt_all_data = np.sum(cummulative_hist * ((bin_center - mean_all_data)/stdev_all_data) ** 4) / np.sum(cummulative_hist) - 3

        if self.plot_data_stats and self.is_global_zero:
            _, axes = plt.subplots(ncols=3, nrows=2, figsize=(9, 4.5))
            axes = axes.flatten()
            bins = 25
            plt.suptitle(f"BES statistics | `{dataset_stage}` dataset with {len(confinement_data)} confinement modes")
            plt.sca(axes[0])
            # plt.hist(
            #     np.array([confinement_mode['pre_elm_size'] for elm in elm_data])/1e3, 
            #     bins=bins,
            # )
            plt.xlabel('size (ms)')
            plt.sca(axes[1])
            plt.hist(
                np.concatenate([confinement_mode['maxabs_by_channel'] for confinement_mode in confinement_data], axis=None), 
                bins=bins,
            )
            plt.xlabel('Channel-wise max(abs())')
            plt.sca(axes[2])
            plt.hist(
                np.concatenate([confinement_mode['maxcount_by_channel'] for confinement_mode in confinement_data], axis=None), 
                bins=20,
                range=(0,20),
            )
            plt.xlabel('Channel-wise saturated points')
            plt.sca(axes[3])
            plt.hist(
                np.concatenate([confinement_mode['mean_by_channel'] for confinement_mode in confinement_data], axis=None), 
                bins=bins,
            )
            plt.xlabel('Channel-wise mean')
            plt.sca(axes[4])
            plt.hist(
                np.concatenate([confinement_mode['std_by_channel'] for confinement_mode in confinement_data], axis=None), 
                bins=bins,
            )
            plt.xlabel('Channel-wise std. dev.')
            plt.sca(axes[5])
            plt.hist(
                np.log10(np.concatenate([confinement_mode['kurt_by_channel'] for confinement_mode in confinement_data], axis=None)),
                bins=bins,
            )
            plt.xlabel('Channel-wise log10(kurt)')
            for axis in axes:
                plt.sca(axis)
                plt.ylabel('Counts')
                plt.yscale('log')
                plt.ylim(bottom=0.8)
            plt.tight_layout()
            filepath = os.path.join(self.log_dir, f'{dataset_stage}_dataset_stats.pdf')
            print(f"  Saving figure {filepath}")
            plt.savefig(filepath, format='pdf', transparent=True)
            plt.show(block=False)

        print(f"  Global min/max raw signal, ch 1-32: {np.amin(packaged_signals[:,:4,:]):.6f}, {np.amax(packaged_signals[:,:4,:]):.6f}")
        print(f"  Global min/max raw signal, ch 33-64: {np.amin(packaged_signals[:,4:,:]):.6f}, {np.amax(packaged_signals[:,4:,:]):.6f}")

        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in confinement_data], axis=0)
        if self.one_hot_labels:
            encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(self.num_classes)], handle_unknown='ignore')
            packaged_labels = encoder.fit_transform(packaged_labels.reshape(-1, 1))

        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in confinement_data], axis=0)
        # assert packaged_labels.size == packaged_valid_t0.size
        # assert packaged_labels.shape[0] == packaged_valid_t0.shape[0]

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
        packaged_shot = np.array(
            [confinement_mode['shot'] for confinement_mode in confinement_data],
            dtype=int,
        )
        packaged_start_time = np.array(
            [confinement_mode['time'] for confinement_mode in confinement_data],
            dtype=int,
        )
        del confinement_data

        # valid t0 indices
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        # assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices + self.signal_window_size]))
        print("  Raw data stats")
        stats = self._get_statistics(
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

            stats = self._get_statistics(
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
            stats = self._get_statistics(
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
            # packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev
            for idx, signal in enumerate(packaged_signals):
                packaged_signals[idx] = (signal - self.signal_mean) / self.signal_stdev
            stats = self._get_statistics(
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
            self.datasets[dataset_stage] = Confinement_TrainValTest_Dataset(
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
            del self.datasets['validation']
            
            predict_datasets = []
            for i_confinement_mode, idx_start in enumerate(packaged_window_start):
                if self.max_predict_confinement_modes and i_confinement_mode == self.max_predict_confinement_modes:
                    break
                if i_confinement_mode == packaged_window_start.size - 1:
                    idx_stop = packaged_labels.size - 1
                else:
                    idx_stop = packaged_window_start[i_confinement_mode+1]-1
                dataset = Confinement_Predict_Dataset(
                    signals=packaged_signals[idx_start:idx_stop, ...],
                    labels=packaged_labels[idx_start:idx_stop],
                    signal_window_size=self.signal_window_size,
                    shot=packaged_shot[i_confinement_mode],
                    start_time=packaged_start_time[i_confinement_mode],
                    confinement_mode_index=packaged_confinement_mode_key[i_confinement_mode],
                )
                predict_datasets.append(dataset)
            self.datasets['predict'] = predict_datasets
            # return predict_datasets
        # print(f"  Data stage `{dataset_stage}` elapsed time {(time.time()-t0)/60:.1f} min")
        gc.collect()
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def reshape_signals_6x8(self, signals, inboard_order):
        # Assumptions:
        # - `inboard_order` contains valid indices for the starting positions of each row.
        # - `signals` is expected to be of shape (num_channels, num_samples), where num_channels >= max(inboard_order) + 7.

        # truncate the inboard_order array to first 6 rows
        inboard_order = inboard_order[:6]

        # Initialize the reshaped signals with zeros or np.nan if there's a chance of not filling some cells
        reshaped_signals = np.zeros((signals.shape[1], 6, 8), dtype=np.float32)  # Using zeros as default values

        # Reshape signals according to the truncated inboard_order
        for row, start_idx in enumerate(inboard_order):
            for col in range(8):
                channel_idx = start_idx + col - 1  # Adjusting for 0-indexing if inboard_order is 1-indexed

                # Ensure the calculated index is within the bounds of the signals array
                if 0 <= channel_idx < signals.shape[0]:
                    reshaped_signals[:, row, col] = signals[channel_idx, :]
                else:
                    print(f"Warning: Channel index {channel_idx} out of bounds for row {row}, col {col}.")

        return reshaped_signals


    def apply_bandpass_filter(self, signals):
            """
            Applies a bandpass filter to the given signals if the cutoff frequencies are specified.
            Otherwise, returns the original signals.

            Args:
                packaged_signals: The signals to be filtered.

            Returns:
                Filtered signals or the original signals.
            """
            required_length = 3 * self.filter_taps  # Set to 3 times the number of filter taps

            # Check if the cutoff frequencies are specified
            if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None and signals.shape[0] > required_length:
                # Design the bandpass filter
                bandpass_filter = firwin(
                    self.filter_taps,
                    [self.lower_cutoff_frequency_hz, self.upper_cutoff_frequency_hz],
                    pass_zero=False,
                    fs=self.sampling_frequency_hz
                )

                # Apply the filter
                filtered_signals = filtfilt(bandpass_filter, 1, signals, axis=0)
                return filtered_signals
            else:
                # print("BANDPASS FILTER NOT APPLIED")
                return signals

    def _get_confinement_events_and_split(self):
        print(f"Data file: {self.data_file}")
        shots = self._load_and_label_shots()
        self._apply_metadata_filters(shots)
        self._apply_forced_shots(shots)
        filtered_shots = self._filter_shots_by_class(shots)
        self._shuffle_and_limit_shots(filtered_shots)
        self._split_datasets(filtered_shots)
        self.calculate_mode_times_and_shots(self.data_file, list(filtered_shots.keys()))

    def _load_and_label_shots(self):
        if self.bad_shots is None:
            self.bad_shots = []  # Initialize to empty list if None

        shots = {}
        with h5py.File(self.data_file, "r") as data_file:
            for shot in data_file.keys():
                if shot in self.bad_shots:
                    print(f"Skipping bad shot: {shot}")
                    continue
                shot_labels = self._collect_labels(data_file, shot)
                if not shot_labels:
                    continue
                label_presence = self._label_presence(shot_labels)
                shot_events = [(shot, event) for event in data_file[shot].keys() if 'labels' in data_file[shot][event]]
                metadata = self._extract_metadata(data_file[shot].attrs)
                shots[shot] = (shot_events, label_presence, metadata)
        return shots

    def _apply_metadata_filters(self, shots):
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        for shot in list(shots):
            metadata = shots[shot][2]
            if not self._metadata_within_bounds(metadata):
                shots.pop(shot)
                if metadata['r_avg'] is None or not self._check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
                    r_avg_exclusions += 1
                if metadata['z_avg'] is None or not self._check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
                    z_avg_exclusions += 1
                if metadata['delz_avg'] is None or not self._check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
                    delz_avg_exclusions += 1
        print(f"Number of r_avg exclusions: {r_avg_exclusions}")
        print(f"Number of z_avg exclusions: {z_avg_exclusions}")
        print(f"Number of delz_avg exclusions: {delz_avg_exclusions}")

    def _metadata_within_bounds(self, metadata):
        return all(self._check_bounds(metadata[key], self.metadata_bounds[key]) for key in ['r_avg', 'z_avg', 'delz_avg'] if key in self.metadata_bounds)

    def _check_bounds(self, value, bounds):
        return bounds[0] <= value <= bounds[1] if bounds else True

    def _extract_metadata(self, attrs):
        return {
            'r_avg': attrs.get('r_avg'),
            'z_avg': attrs.get('z_avg'),
            'delz_avg': attrs.get('delz_avg')
        }
    
    def _apply_forced_shots(self, shots):
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

         # Return shots including forced shots for further processing
        return {**shots, **test_shot_data, **validation_shot_data}

    def _collect_labels(self, data_file, shot):
        shot_labels = []
        for event in data_file[shot].keys():
            event_data = data_file[shot][event]
            if 'labels' in event_data:
                labels_array = event_data['labels'][()]
                shot_labels.extend(labels_array)
        return shot_labels

    def _label_presence(self, labels):
        return tuple(class_id in labels for class_id in range(self.num_classes))  

    def _filter_shots_by_class(self, shots):
        shots_by_class = {}
        for shot, (events, labels, metadata) in shots.items():
            if labels not in shots_by_class:
                shots_by_class[labels] = []
            shots_by_class[labels].append(shot)
        if self.max_shots_per_class is not None:
            for labels, shot_list in shots_by_class.items():
                if len(shot_list) > self.max_shots_per_class:
                    shots_by_class[labels] = np.random.choice(shot_list, self.max_shots_per_class, replace=False).tolist()
        return {shot: shots[shot] for label_shots in shots_by_class.values() for shot in label_shots}

    def _shuffle_and_limit_shots(self, filtered_shots):
        shot_numbers = np.array(list(filtered_shots.keys()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(shot_numbers)
        if self.max_shots:
            shot_numbers = shot_numbers[:self.max_shots]
        self.all_confinement_events = np.concatenate([filtered_shots[shot][0] for shot in shot_numbers])

    def _split_datasets(self, filtered_shots):
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


    def map_labels(self, label):
        # Define less common labels
        uncommon_labels_0 = [
            (True, True, False, False),
            (False, True, True, False),
            (True, False, True, True),
            (False, False, True, True), 
            # (True, True, True, False), # Add other similar rare labels if needed
        ]
        uncommon_labels_1 = [
            (True, False, True, False),  # Add other similar rare labels if needed
        ]
        # Map less common labels to a new, single label
        if label in uncommon_labels_0:
            return (True, True, True, True)  # Example: re-map to this label
        elif label in uncommon_labels_1:
            return (True, False, False, False)
        return label

    def calculate_mode_times_and_shots(self, file_path, valid_shot_keys):
        mode_times = {}  # Time spent in each mode
        mode_shots = {}  # Unique shots for each mode

        with h5py.File(file_path, 'r') as file:
            for shot_key in valid_shot_keys:
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
    
    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Determine valid t0 indices (start of signal windows) for real-time inference
        valid_t0 = np.zeros(labels.size, dtype=int)
        first_valid_signal_window_start_index = self.signal_window_size - 1
        valid_t0[first_valid_signal_window_start_index:] = 1

        # if self.log_time:
        #     labels = np.log10(labels)

        return labels, valid_t0
    
    def _get_statistics(
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

    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        valid_sampler = torch.utils.data.DistributedSampler(
            self.datasets['validation'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=valid_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        ) 
                
    def test_dataloader(self):
        test_sampler = torch.utils.data.DistributedSampler(
            self.datasets['test'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            sampler=test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        ) 
    
    def predict_dataloader(self):
        predict_sampler = torch.utils.data.DistributedSampler(
            self.datasets['test'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            sampler=predict_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            # pin_memory=False,
            persistent_workers=False,
        ) 
