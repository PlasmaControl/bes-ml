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

class TrainValTest_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            signals: np.ndarray,       # shape (total_time_length, n_rows, n_cols)
            n_rows: int,
            n_cols: int,
            labels: np.ndarray,        # shape (total_time_length, 8)
            sample_indices: np.ndarray,
            signal_window_size: int,
            time_points: np.ndarray    # shape (total_time_length,)
    ) -> None:
        # Add channel dimension to signals
        self.signals = torch.from_numpy(np.ascontiguousarray(signals)[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == n_rows and
            self.signals.size(3) == n_cols
        ), "Signals have incorrect shape"

        self.labels = torch.from_numpy(labels)    # shape (total_time_length, 8)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.sample_indices = torch.from_numpy(sample_indices)
        self.signal_window_size = signal_window_size
        self.time_points = torch.from_numpy(time_points)

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, idx: int) -> tuple:
        i_t0 = self.sample_indices[idx]
        start_index = i_t0 - self.signal_window_size + 1
        end_index = i_t0 + 1

        # Adjust for flattened structure
        signal_index = i_t0 // self.n_cols  # Extract the time index
        col_index = i_t0 % self.n_cols      # Extract the column index

        if start_index < 0 or end_index > self.signals.shape[1]:
            raise IndexError(f"Invalid indices: start_index={start_index}, end_index={end_index}, signal length={self.signals.shape[1]}")

        # Extract the signal window for the correct column
        signal_window = self.signals[:, start_index:end_index, col_index]  # Shape: (window_size, n_rows)

        # Get label and radial position (if available)
        label = self.labels[signal_index * self.n_cols + col_index]
        radial_position = self.radial_positions[col_index] if self.radial_positions is not None else None

        time_point = self.time_points[signal_index]

        return signal_window, label, time_point

class TrainValTest_Dataset_3(torch.utils.data.Dataset):
    def __init__(
            self,
            signals: np.ndarray,       # shape (total_time * n_cols, n_rows)
            n_rows: int,
            n_cols: int,
            labels: np.ndarray,        # shape (total_time * n_cols,)
            sample_indices: np.ndarray,
            signal_window_size: int,
            time_points: np.ndarray,   # shape (total_time * n_cols,)
            radial_positions: np.ndarray = None  # shape (n_cols,), optional
    ) -> None:
        # Add channel dimension to signals
        self.signals = torch.from_numpy(np.ascontiguousarray(signals)[np.newaxis, ...]).float()
        assert (
            self.signals.ndim == 3 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == n_rows
        ), "Signals have incorrect shape"

        self.labels = torch.from_numpy(labels).float()    # Shape (total_samples,)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.sample_indices = torch.from_numpy(sample_indices).long()
        self.signal_window_size = signal_window_size
        self.time_points = torch.from_numpy(time_points).float()
        self.radial_positions = torch.from_numpy(radial_positions).float() if radial_positions is not None else None

        assert self.signals.shape[1] == self.labels.shape[0], "Mismatch between signals and labels count"

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, idx: int) -> tuple:
        i_t0 = self.sample_indices[idx]
        start_index = i_t0 - self.signal_window_size + 1
        end_index = i_t0 + 1

        # Adjust for flattened structure and validate indices
        if start_index < 0 or end_index > self.signals.shape[1]:
            raise IndexError(f"Invalid indices: start_index={start_index}, end_index={end_index}, signal length={self.signals.shape[1]}")

        # Extract the signal window
        signal_window = self.signals[:, start_index:end_index, :]  # Shape: (1, window_size, n_rows)

        # Get label and radial position (if available)
        label = self.labels[i_t0]
        radial_position = self.radial_positions[i_t0 % self.n_cols] if self.radial_positions is not None else None
        time_point = self.time_points[i_t0 // self.n_cols]

        return signal_window, label, time_point

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Args:
            data: List of (signal_window, label, time_point, shot_id) tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal_window, label, time_point, shot_id = self.data[idx]
        return (
            torch.tensor(signal_window, dtype=torch.float32).unsqueeze(0),  # Add channel dim
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(time_point, dtype=torch.float32),
            shot_id,  # Keep shot_id as it is
        )
    
class PredictDataset_4(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Args:
            data: List of (signal_window, label, time_point, shot_id, radial_position) tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal_window, label, time_point, shot_id, radial_position = self.data[idx]
        # Return radial_position as a single float tensor
        return (
            torch.tensor(signal_window, dtype=torch.float32).unsqueeze(0),  
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(time_point, dtype=torch.float32),
            shot_id,  # shot_id as is (string/int)
            torch.tensor(radial_position, dtype=torch.float32)
        )
            
@dataclasses.dataclass(eq=False)
class Velocimetry_Datamodule(LightningDataModule):
    data_file: str = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/191376_restructure_velocimetry.hdf5'  # path to data; dir or file depending on task
    n_rows: int = 8
    n_cols: int = 8
    batch_size: int = 128  # power of 2, like 32-256
    signal_window_size: int = 8  # power of 2, like 64-512
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    seed: int = 0  # RNG seed for deterministic, reproducible shuffling of events
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }
    split_method: str = 'event'    
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    test_only: bool = False
    max_shots: int = None
    max_events: int = None
    max_shots_per_class: int = None
    num_classes: int = 4
    max_predict_events: int = 24
    bad_shots: list = None
    train_shots: list = None
    validation_shots: list = None
    test_shots: list = None
    predict_shots: list = None
    force_validation_shots: list = None
    force_test_shots: list = None
    # Bandpass filter parameters
    sampling_frequency_hz: float = 1 / 10**(-6)  # Sampling frequency in Hz
    filter_taps: int = 501  # Number of taps in the filter
    lower_cutoff_frequency_hz: float = None  # Lower cutoff frequency in Hz
    upper_cutoff_frequency_hz: float = None  # Upper cutoff frequency in Hz
    highpass_cutoff_frequency_hz: float = None # Lower cutoff frequency in Hz (typically used after downsampling BES signals)
    signal_mean: float = None  
    signal_stdev: float = None  
    start_time_ms: float = None
    end_time_ms: float = None
    clip_signals: float = None # remove signal windows with abs(raw_signals) > clip_signals
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    target_labels: list[str] = dataclasses.field(default_factory=lambda: ["vZ", "vZ_uncertainty"]) 
    ignored_columns: list[int] = dataclasses.field(default_factory=list)
    clip_labels: bool = False
    labels_lower_bound: float = None
    labels_upper_bound: float = None
    standardize_labels: bool = True  # Option to standardize labels
    normalize_labels: bool = False   # Option to normalize labels
    label_mean: float = 0.0           # mean value for standardization
    label_std: float = 1.0            # std dev value for standardization
    label_min: float = -2e9           # min value for normalization
    label_max: float = 2e9            # max value for normalization
    vZ_uncertainty_threshold: float = 200.0
    do_flip_augmentation: bool = False  
    split_train_data_per_gpu: bool = True  
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
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
        self.all_events = None
        self.test_events = None
        self.train_events = None
        self.validation_events = None
        self.predict_events = None
        self._get_events_and_split()
        self.dataset_events = {
                'train': self.train_events,
                'validation': self.validation_events,
                'test': self.test_events,
                'predict': self.predict_events,
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
        print(f"Running Velocimetry_Datamodule.setup(stage={stage})")
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
        if self.split_train_data_per_gpu:
            for dataset_stage in dataset_stages:
                # Determine the chunk of confinement indices for this GPU
                events = self.dataset_events[dataset_stage]
                times = [self.get_time_for_index(shot_event) for shot_event in events]  # Adapted for (shot, event) tuples
                if dataset_stage in ['train']:
                    print(f"Creating chunks for {dataset_stage} with {len(events)} indices and total time {sum(times)}")
                    # Create balanced chunks
                    chunks = self.create_balanced_chunks(events, times, self.world_size)
                    # Determine the chunk for this GPU
                    chunk_events = chunks[global_rank]
                elif dataset_stage in ['validation', 'test', 'predict']:
                    chunk_events = events

                if dataset_stage in ['train', 'validation', 'test']:
                    dataset = self._load_and_preprocess_data_4(chunk_events, dataset_stage)
                elif dataset_stage in ['predict']:
                    print(f"Preparing predict dataset for shots: {self.predict_shots}")
                    # Load and preprocess the data for these events
                    dataset = self._load_and_preprocess_predict_data_4(chunk_events)
                    # Assign the predict dataset
                    self.datasets["predict"] = dataset

                # Store the DataLoader for this GPU
                if dataset_stage == 'train':
                    self._train_dataloader = torch.utils.data.DataLoader(
                                            dataset, 
                                            batch_size=self.batch_size,
                                            shuffle=True,             
                                            num_workers=self.num_workers,
                                            persistent_workers=(self.num_workers > 0),
                                            drop_last=True,
                                            # collate_fn=self.custom_collate_fn,
                                            )
        else:
            for dataset_stage in dataset_stages:
                events = self.dataset_events[dataset_stage]

                if dataset_stage in ['train', 'validation', 'test']:
                    dataset = self._load_and_preprocess_data_4(events, dataset_stage)
                elif dataset_stage in ['predict']:
                    # Load and preprocess the data for these events
                    dataset = self._load_and_preprocess_predict_data_4(chunk_events)
                    # Assign the predict dataset
                    self.datasets["predict"] = dataset

         
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

        # return chunks, [index_to_time[index] for index in indices]
        return chunks
        
    def _load_and_preprocess_data(self, shot_event_indices, dataset_stage):
        t0 = time.time()
        print(f"Reading velocimetry events for dataset `{dataset_stage}`")
        velocimetry_data = []
        n_bins = 201
        cummulative_hist = np.zeros(n_bins, dtype=int)

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

            time_count = int(np.sum(time_counts))
            discarded_count = len(shot_event_indices) - len(long_enough_indices)
            print(f"Discarded {discarded_count} events due to insufficient signal length or missing inboard order.")
            
            packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
            packaged_labels = {key: [] for key in ["vZ_shear_profile"]}
            start_index = 0
            for i, (shot, event) in enumerate(long_enough_indices):
                if i % 10 == 0:
                    print(f"  Reading event {i:04d}/{len(shot_event_indices):04d}, event: {event} in shot {shot}")
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Retrieve signals and reshape according to inboard_order
                signals = np.array(event_data["signals"][:, :], dtype=np.float32)
                signals = self.reshape_signals_8x8(signals, inboard_order) # shape (T, 8, 8)
                start_col_index = (8-self.n_cols)
                signals = signals[:, :self.n_rows, start_col_index:] # take first n rows and take last m columns of array optionally
                
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = self.apply_bandpass_filter(signals)

                if self.highpass_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying highpass filter ")
                    signals = self.apply_highpass_filter(signals)
                
                # Retrieve labels shape: (T, 8)
                vZ_shear_profile = np.array(event_data["flow_shear"][:, :], dtype=np.float32)

                # Determine the shortest time dimension across all arrays
                min_time_length = min(signals.shape[0], vZ_shear_profile.shape[0])

                # Truncate all arrays to the shortest time dimension
                signals = signals[:min_time_length]
                vZ_shear_profile = vZ_shear_profile[:min_time_length]

                assert signals.shape[0] == vZ_shear_profile.shape[0], "signals and labels must have same time dimension"

                packaged_signals[start_index:start_index + signals.shape[0]] = signals
                start_index += signals.shape[0]

                # Compute valid_t0 for the current event
                valid_t0 = self._get_valid_indices(vZ_shear_profile)

                velocimetry_data.append({
                    'valid_t0': valid_t0,
                    'vZ_shear_profile': vZ_shear_profile,
                    'event_key': event_key,
                    'shot': shot,
                    'time': event,
                    'time_data': np.array(event_data["times"][:min_time_length], dtype=np.float32),
                })

        # print(f"  Global min/max raw signal, ch 1-32: {np.amin(packaged_signals[:,:4,:]):.6f}, {np.amax(packaged_signals[:,:4,:]):.6f}")
        # print(f"  Global min/max raw signal, ch 33-64: {np.amin(packaged_signals[:,4:,:]):.6f}, {np.amax(packaged_signals[:,4:,:]):.6f}")
        # Only consider the populated part of packaged_signals
        populated_signals = packaged_signals[:start_index]

        # Compute min/max for populated signals
        print(f"  Global min/max raw signal, ch 1-32: {np.amin(populated_signals[:,:4,:]):.6f}, {np.amax(populated_signals[:,:4,:]):.6f}")
        print(f"  Global min/max raw signal, ch 33-64: {np.amin(populated_signals[:,4:,:]):.6f}, {np.amax(populated_signals[:,4:,:]):.6f}")

        # concatenate events
        packaged_vZ_shear_profile = np.concatenate([event['vZ_shear_profile'] for event in velocimetry_data], axis=0)
        packaged_valid_t0 = np.concatenate([event['valid_t0'] for event in velocimetry_data], axis=0)
        packaged_times = np.concatenate([event['time_data'] for event in velocimetry_data], axis=0)

        # store labels in packaged_labels dict
        packaged_labels["vZ_shear_profile"] = packaged_vZ_shear_profile

        # start indices for each event in concatenated dataset
        packaged_window_start = []
        index = 0
        for event in velocimetry_data:
            packaged_window_start.append(index)
            index += event['time_data'].size
        packaged_window_start = np.array(packaged_window_start, dtype=int)

        packaged_event_key = np.array(
            [event['event_key'] for event in velocimetry_data],
            dtype=str,
        )
        packaged_shot = np.array(
            [event['shot'] for event in velocimetry_data],
            dtype=int,
        )
        packaged_start_time = np.array(
            [event['time'] for event in velocimetry_data],
            dtype=int,
        )
        del velocimetry_data

        # valid t0 indices
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        print(f"Filtered valid indices count: {packaged_valid_t0_indices.size}")

        assert all(np.all(np.isfinite(packaged_labels[key][packaged_valid_t0_indices])) for key in packaged_labels.keys())

        print("  Raw data stats")
        stats = self._get_statistics(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

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
            assert dataset_stage == 'train' or not self.train_events, f"Dataset_stage: {dataset_stage}"
            print(f"  Calculating signal mean and std from {dataset_stage} data")
            self.signal_mean = stats['mean']
            self.signal_stdev = stats['stdev']
            self.signal_exkurt = stats['exkurt']
            self.save_hyperparameters({
                'signal_mean': self.signal_mean.item(),
                'signal_stdev': self.signal_stdev.item(),
                'signal_exkurt': self.signal_exkurt.item(),
            })

        if dataset_stage in ['train', 'validation', 'test']:
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
            dataset = TrainValTest_Dataset(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                window_start_indices=packaged_window_start,
                signal_window_size=self.signal_window_size,
                event_keys=packaged_event_key,
                time_points=packaged_times,

            )
            return dataset
        if dataset_stage in ['validation', 'test']:
            self.datasets[dataset_stage] = TrainValTest_Dataset(
                    signals=packaged_signals,
                    n_rows=self.n_rows,
                    n_cols=self.n_cols,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    window_start_indices=packaged_window_start,
                    signal_window_size=self.signal_window_size,
                    event_keys=packaged_event_key,
                    time_points=packaged_times,
                )
            return
        
        gc.collect()
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _load_and_preprocess_data_2(self, shot_start_time_indices, dataset_stage):

        t0 = time.time()
        print(f"Reading data for dataset `{dataset_stage}`")

        labels_list = []
        valid_t0_indices_list = []  # Changed from valid_t0_list to valid_t0_indices_list
        times_list = []
        event_keys_list = []
        shots_list = []
        start_times_list = []

        total_time_length = 0  # To adjust indices when concatenating

        with h5py.File(self.data_file, 'r') as h5_file:
            if len(shot_start_time_indices) >= 5:
                print(f"  Initial shot/event indices: {shot_start_time_indices[:5]}")
            time_counts = []
            long_enough_indices = []  # List to hold indices of events with long enough signals
            for i, (shot, event) in enumerate(shot_start_time_indices):
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

            time_count = int(np.sum(time_counts))
            discarded_count = len(shot_start_time_indices) - len(long_enough_indices)
            print(f"Discarded {discarded_count} events due to insufficient signal length or missing inboard order.")
            
            packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
            signals_start_index = 0

            for i, (shot, start_time) in enumerate(long_enough_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(long_enough_indices):04d}, start_time: {start_time} in shot {shot}")
                event_key = f"{shot}/{int(start_time)}"
                event_data = h5_file[event_key]

                # Read signals and times
                signals = np.array(event_data["signals"][:, :], dtype=np.float32)  # shape (64, time)
                times = np.array(event_data["times"][:], dtype=np.float32)  # shape (time,)
                
                # Read labels and label_times
                label_times = np.array(event_data["label_times"][:], dtype=np.float32)  # shape (label_times,)
                target = np.array(event_data["vZ"][:, :], dtype=np.float32)  # shape (label_times, 8)
                
                # Map signal times to label indices
                label_indices = np.searchsorted(label_times, times, side='left')
                label_indices[label_indices >= len(label_times)] = len(label_times) - 1
                mask = (label_indices > 0) & (
                    np.abs(times - label_times[label_indices - 1]) < np.abs(times - label_times[label_indices])
                )
                label_indices[mask] -= 1

                # Align labels with signal times
                labels_aligned = target[label_indices, :self.n_cols]  # shape (time, n_cols)

                if self.clip_labels:
                    if i % 100 == 0:
                        print(f"    Clipping labels...")
                    # Apply label filtering
                    valid_mask = (labels_aligned >= self.labels_lower_bound) & (labels_aligned <= self.labels_upper_bound)

                    # For multi-dimensional labels, combine masks across columns
                    valid_mask = np.all(valid_mask, axis=1)
                else:
                    # When not clipping, all data is considered valid
                    valid_mask = np.ones(len(labels_aligned), dtype=bool)

                # Filter signals and labels
                signals = signals[:, valid_mask]
                labels_aligned = labels_aligned[valid_mask]
                times = times[valid_mask]

                if signals.shape[1] == 0:
                    if i % 10 == 0:
                        print(f"signals length 0 after clipping labels, skipping event")
                    continue

                mean = np.mean(signals, axis=1, keepdims=True)
                std = np.std(signals, axis=1, keepdims=True)
                
                # Avoid division by zero by setting std to 1 where it is zero
                std[std == 0] = 1
                
                # Standardize signals
                signals = (signals - mean) / std

                # Apply label transformations
                if self.standardize_labels:
                    if i % 100 == 0:
                        print(f"    Standardizing labels...")
                    labels_aligned = (labels_aligned - self.label_mean) / self.label_std

                elif self.normalize_labels:
                    if i % 100 == 0:
                        print(f"    Normalizing labels to range [-1, 1]...")
                    labels_aligned = -1 + 2 * (labels_aligned - self.label_min) / (self.label_max - self.label_min)

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Reshape signals to (time, n_rows, n_cols)
                signals = self.reshape_signals(signals, inboard_order)

                assert signals.shape[0] == labels_aligned.shape[0] == times.shape[0], 'signals, labels, time do not have same time dimension'

                # Compute valid_t0 for the current event
                valid_t0 = self._get_valid_indices(labels_aligned)

                # Compute valid_t0_indices for the event
                event_valid_t0_indices = np.where(valid_t0 == 1)[0]

                # Adjust indices to account for concatenation
                valid_t0_indices = event_valid_t0_indices + total_time_length

                # Append data to lists
                labels_list.append(labels_aligned)
                valid_t0_indices_list.append(valid_t0_indices)
                times_list.append(times)
                event_keys_list.append(event_key)
                shots_list.append(shot)
                start_times_list.append(start_time)

                total_time_length += len(times)
                packaged_signals[signals_start_index:signals_start_index + signals.shape[0]] = signals
                signals_start_index += signals.shape[0]

        # Concatenate data from all events
        packaged_labels = np.concatenate(labels_list, axis=0)    # shape (total_time_length, 8)
        # packaged_valid_t0 = np.concatenate(valid_t0_list)
        packaged_times = np.concatenate(times_list)

        packaged_event_key = np.array(event_keys_list)
        packaged_shot = np.array(shots_list)
        packaged_start_time = np.array(start_times_list)

        # Concatenate valid_t0_indices from all events
        packaged_valid_t0_indices = np.concatenate(valid_t0_indices_list)

        # Ensure indices are within bounds
        signals_length = packaged_signals.shape[0]
        packaged_valid_t0_indices = packaged_valid_t0_indices[
            (packaged_valid_t0_indices >= self.signal_window_size - 1) &
            (packaged_valid_t0_indices + 1 <= signals_length)
        ]
        print(f"Filtered valid indices count: {packaged_valid_t0_indices.size}")

        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))

        print("  Raw data stats")
        stats = self._get_statistics(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

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
        # if None in [self.signal_mean, self.signal_stdev]:
        #     assert dataset_stage == 'train' or not self.train_events, f"Dataset_stage: {dataset_stage}"
        #     print(f"  Calculating signal mean and std from {dataset_stage} data")
        #     self.signal_mean = stats['mean']
        #     self.signal_stdev = stats['stdev']
        #     self.signal_exkurt = stats['exkurt']
        #     self.save_hyperparameters({
        #         'signal_mean': self.signal_mean.item(),
        #         'signal_stdev': self.signal_stdev.item(),
        #         'signal_exkurt': self.signal_exkurt.item(),
        #     })

        # if dataset_stage in ['train', 'validation', 'test']:
        #     print(f"  Standarizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f}")
        #     print(f"  Standardized signal stats")
        #     # packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev
        #     for idx, signal in enumerate(packaged_signals):
        #         packaged_signals[idx] = (signal - self.signal_mean) / self.signal_stdev
        #     stats = self._get_statistics(
        #         sample_indices=packaged_valid_t0_indices,
        #         signals=packaged_signals,
        #     )
        # self.max_abs_valid_signal = np.max(np.abs([stats['min'],stats['max']]))
            
        if dataset_stage in ['train']:
            dataset = TrainValTest_Dataset(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                signal_window_size=self.signal_window_size,
                time_points=packaged_times,
            )
            return dataset
        if dataset_stage in ['validation', 'test']:
            self.datasets[dataset_stage] = TrainValTest_Dataset(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                signal_window_size=self.signal_window_size,
                time_points=packaged_times,
            )
            return
        
        gc.collect()
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _load_and_preprocess_data_3(self, shot_start_time_indices, dataset_stage):
        """
        Load and preprocess BES data for the machine learning pipeline.
        Processes signals column-wise, applies masking for high uncertainty, and integrates radial positions.
        """
        t0 = time.time()
        print(f"Reading data for dataset {dataset_stage}")

        signals_list = []
        labels_list = []
        times_list = []
        radial_positions_list = []
        valid_t0_list = []

        with h5py.File(self.data_file, 'r') as h5_file:
            if len(shot_start_time_indices) >= 5:
                print(f"  Initial shot/event indices: {shot_start_time_indices[:5]}")

            time_counts = []
            long_enough_indices = []

            for i, (shot, event) in enumerate(shot_start_time_indices):
                event_key = f"{shot}/{event}"
                signal_length = h5_file[event_key]["signals"].shape[1]

                if signal_length >= self.signal_window_size:
                    inboard_order = h5_file[shot].attrs.get("inboard_column_channel_order", None)

                    if inboard_order is None or len(inboard_order) == 0:
                        print(f"Skipping event {event_key} due to missing or empty inboard_column_channel_order.")
                        continue

                    time_counts.append(signal_length)
                    long_enough_indices.append((shot, event))

            time_count = int(np.sum(time_counts))
            discarded_count = len(shot_start_time_indices) - len(long_enough_indices)
            print(f"Discarded {discarded_count} events due to insufficient signal length or missing inboard order.")

            for i, (shot, start_time) in enumerate(long_enough_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(long_enough_indices):04d}, start_time: {start_time} in shot {shot}")

                event_key = f"{shot}/{int(start_time)}"
                event_data = h5_file[event_key]

                signals = np.array(event_data["signals"], dtype=np.float32)  # shape (64, time)
                times = np.array(event_data["times"], dtype=np.float32)      # shape (time,)
                label_times = np.array(event_data["label_times"], dtype=np.float32)  # shape (label_times,)
                target = np.array(event_data["vZ"], dtype=np.float32)        # shape (label_times, n_cols)
                uncertainty = np.array(event_data["vZ_uncertainty"], dtype=np.float32)  # shape (label_times, n_cols)

                # Reshape raw signals to (time, n_rows, n_cols)
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]
                signals = self.reshape_signals(signals, inboard_order)       # Shape now (time, n_rows, n_cols)

                # Retrieve radial positions
                r_position = h5_file[shot].attrs["r_position"]
                if r_position is not None and len(r_position) == 64:
                    radial_positions = r_position.reshape(8, 8)[0, :]  # Extract unique radial positions (first row)
                else:
                    print(f"Warning: Radial positions missing or malformed for shot {shot}.")
                    radial_positions = np.zeros(self.n_cols)  # Default placeholder

                # Filter and transform labels
                labels_aligned = self._filter_and_transform_labels(label_times, target, uncertainty, times)

                # Skip if all labels are NaN
                if np.isnan(labels_aligned).all():
                    continue

                # Remove NaN labels along the time axis
                signals = signals[~np.isnan(labels_aligned).any(axis=1), :, :]
                labels_aligned = labels_aligned[~np.isnan(labels_aligned).any(axis=1), :]

                for col in range(self.n_cols):
                    col_signals = signals[:, :, col]  # Extract individual column signals
                    col_labels = labels_aligned[:, col]

                    # Normalize and standardize signals
                    # mean = np.mean(col_signals, axis=1, keepdims=True)
                    # std = np.std(col_signals, axis=1, keepdims=True)
                    # std[std == 0] = 1  # Avoid division by zero
                    # col_signals = (col_signals - mean) / std
                    
                    if col_signals.size == 0 or col_labels.size == 0:
                        continue

                    # Adjust sample indices for this column
                    valid_t0 = np.zeros(times.shape[0], dtype=int)
                    valid_t0[self.signal_window_size - 1:] = 1  # Mark valid t0 after the first sws-1 points

                    # Append flattened signals and valid_t0 for this column
                    signals_list.append(col_signals)  # Shape: (time, rows)
                    labels_list.append(col_labels)
                    valid_t0_list.append(valid_t0)   # Shape: (time,)
                    radial_positions_list.append(np.full(col_labels.shape, radial_positions[col]))
                    times_list.append(times)


        # Combine signals and labels from all events
        assert all(s.shape[1] == signals_list[0].shape[1] for s in signals_list), \
            "Inconsistent shapes in signals_list along axis 1"
        
        packaged_signals = np.concatenate(signals_list, axis=0)  # shape (total_time * n_cols, n_rows)
        packaged_labels = np.concatenate(labels_list, axis=0)    # shape (total_time * n_cols,)
        packaged_radial_positions = np.concatenate(radial_positions_list, axis=0)  # shape (total_time * n_cols,)
        packaged_times = np.concatenate(times_list)
        packaged_valid_t0 = np.concatenate(valid_t0_list)

        print(f"Packaged signals shape: {packaged_signals.shape}")
        print(f"Packaged labels shape: {packaged_labels.shape}")
        print(f"Packaged valid t0 shape: {packaged_valid_t0.shape}")
        print(f"Packaged radial positions shape: {packaged_radial_positions.shape}")

        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        # Ensure indices can form a full signal window
        valid_mask = (packaged_valid_t0_indices >= (self.signal_window_size - 1)) & \
                    (packaged_valid_t0_indices < packaged_signals.shape[0])
        packaged_valid_t0_indices = packaged_valid_t0_indices[valid_mask]



        print("  Raw data stats")
        stats = self._get_statistics_2D(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

        # Standardize signals based on training data
        if None in [self.signal_mean, self.signal_stdev]:
            assert dataset_stage == 'train' or not self.train_events, f"Dataset_stage: {dataset_stage}"
            print(f"  Calculating signal mean and std from {dataset_stage} data")
            self.signal_mean = stats['mean']
            self.signal_stdev = stats['stdev']
            self.signal_exkurt = stats['exkurt']
            self.save_hyperparameters({
                'signal_mean': self.signal_mean.item(),
                'signal_stdev': self.signal_stdev.item(),
                'signal_exkurt': self.signal_exkurt.item(),
            })

        if dataset_stage in ['train', 'validation', 'test']:
            print(f"  Standardizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f}")
            packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev  # Standardize all signals
            stats = self._get_statistics_2D(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )

        self.max_abs_valid_signal = np.max(np.abs([stats['min'], stats['max']]))

        if dataset_stage in ['train', 'validation', 'test']:
            self.datasets[dataset_stage] = TrainValTest_Dataset_3(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                signal_window_size=self.signal_window_size,
                time_points=packaged_times,
                radial_positions=packaged_radial_positions
            )
            return
        
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _load_and_preprocess_data_4(self, shot_start_time_indices, dataset_stage):
        """
        Load and preprocess BES data for the machine learning pipeline.
        Processes signals column-wise, applies masking for high uncertainty, and integrates radial positions.
        Now the vZ, vZ_uncertainty, label_times, etc. are saved on the shot level.
        """
        t0 = time.time()
        print(f"Reading data for dataset {dataset_stage}")

        signals_list = []
        labels_list = []
        times_list = []
        radial_positions_list = []
        valid_t0_list = []

        with h5py.File(self.data_file, 'r') as h5_file:
            if len(shot_start_time_indices) >= 5:
                print(f"  Initial shot/event indices: {shot_start_time_indices[:5]}")

            time_counts = []
            long_enough_indices = []

            for i, (shot, event) in enumerate(shot_start_time_indices):
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Check if the event contains the required datasets
                if 'signals' not in event_data or 'times' not in event_data:
                    print(f"{event_key} is missing 'signals' or 'times'.")
                    continue

                signal_length = event_data["signals"].shape[1]
                if signal_length <= self.signal_window_size:
                    print(f"Skipping event {event_key} due to insufficient signal length.")
                    continue         

                inboard_order = h5_file[shot].attrs.get("inboard_column_channel_order", None)

                if inboard_order is None or len(inboard_order) == 0:
                    print(f"Skipping event {event_key} due to missing or empty inboard_column_channel_order.")
                    continue

                time_counts.append(signal_length)
                long_enough_indices.append((shot, event))

            time_count = int(np.sum(time_counts))
            discarded_count = len(shot_start_time_indices) - len(long_enough_indices)
            print(f"Discarded {discarded_count} events due to insufficient signal length or missing inboard order.")

            for i, (shot, start_time) in enumerate(long_enough_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(long_enough_indices):04d}, start_time: {start_time} in shot {shot}")

                # Load event-level data
                event_key = f"{shot}/{start_time}"
                event_data = h5_file[event_key]
                signals = np.array(event_data["signals"], dtype=np.float32)  # shape (64, time)
                times = np.array(event_data["times"], dtype=np.float32)      # shape (time,)
                signal_length = signals.shape[1]

                if signal_length < self.signal_window_size:
                    continue

                # Filter times based on min_time_ms / max_time_ms
                time_mask = np.ones_like(times, dtype=bool)
                if self.start_time_ms is not None:
                    time_mask &= (times >= self.start_time_ms)
                if self.end_time_ms is not None:
                    time_mask &= (times <= self.end_time_ms)
                    # If after masking we have too few data points, skip
                    if times.size < self.signal_window_size:
                        continue

                # Load shot-level data
                shot_group = h5_file[shot]
                label_times = np.array(shot_group["label_times"], dtype=np.float32)  # shape (label_times,)
                target = np.array(shot_group[self.target_labels[0]], dtype=np.float32)
                uncertainty = np.array(shot_group[self.target_labels[1]], dtype=np.float32)  # shape (label_times, n_cols)

                # Reshape raw signals to (time, n_rows, n_cols)
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                signals = self.reshape_signals(signals, inboard_order)       # Shape now (time, n_rows, n_cols)

                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = self.apply_bandpass_filter(signals)

                # Retrieve radial positions
                r_position = h5_file[shot].attrs["r_position"]
                if r_position is not None and len(r_position) == 64:
                    radial_positions = r_position.reshape(8, 8)[0, :]  # Extract unique radial positions (first row)
                else:
                    print(f"Warning: Radial positions missing or malformed for shot {shot}.")
                    radial_positions = np.zeros(self.n_cols)  # Default placeholder

                # Filter and transform labels
                labels_aligned = self._filter_and_transform_labels(label_times, target, uncertainty, times)

                # Skip if all labels are NaN
                if np.isnan(labels_aligned).all():
                    continue

                # Remove NaN labels along the time axis
                signals = signals[~np.isnan(labels_aligned).any(axis=1), :, :]
                labels_aligned = labels_aligned[~np.isnan(labels_aligned).any(axis=1), :]

                for col in range(self.n_cols):

                    if col in self.ignored_columns:
                        continue

                    col_signals = signals[:, :, col]  # Extract individual column signals
                    col_labels = labels_aligned[:, col]
                    
                    if col_signals.size == 0 or col_labels.size == 0:
                        continue

                    # Adjust sample indices for this column
                    valid_t0 = np.zeros(times.shape[0], dtype=int)
                    valid_t0[self.signal_window_size - 1:] = 1  # Mark valid t0 after the first sws-1 points

                    # Append flattened signals and valid_t0 for this column
                    signals_list.append(col_signals)  # Shape: (time, rows)
                    labels_list.append(col_labels)
                    valid_t0_list.append(valid_t0)   # Shape: (time,)
                    radial_positions_list.append(np.full(col_labels.shape, radial_positions[col]))
                    times_list.append(times)

                    if self.do_flip_augmentation:
                        col_signals_flipped = col_signals[:, ::-1]  # Flip rows
                        col_labels_flipped = -col_labels            # Reverse sign

                        signals_list.append(col_signals_flipped)
                        labels_list.append(col_labels_flipped)
                        valid_t0_list.append(valid_t0)  # same time dimension
                        radial_positions_list.append(np.full(col_labels.shape, radial_positions[col]))
                        times_list.append(times)


        # Combine signals and labels from all events
        assert all(s.shape[1] == signals_list[0].shape[1] for s in signals_list), \
            "Inconsistent shapes in signals_list along axis 1"
        
        packaged_signals = np.concatenate(signals_list, axis=0)  # shape (total_time * n_cols, n_rows)
        packaged_labels = np.concatenate(labels_list, axis=0)    # shape (total_time * n_cols,)
        packaged_radial_positions = np.concatenate(radial_positions_list, axis=0)  # shape (total_time * n_cols,)
        packaged_times = np.concatenate(times_list)
        packaged_valid_t0 = np.concatenate(valid_t0_list)

        print(f"Packaged signals shape: {packaged_signals.shape}")
        print(f"Packaged labels shape: {packaged_labels.shape}")
        print(f"Packaged valid t0 shape: {packaged_valid_t0.shape}")
        print(f"Packaged radial positions shape: {packaged_radial_positions.shape}")

        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        # Ensure indices can form a full signal window
        valid_mask = (packaged_valid_t0_indices >= (self.signal_window_size - 1)) & \
                    (packaged_valid_t0_indices < packaged_signals.shape[0])
        packaged_valid_t0_indices = packaged_valid_t0_indices[valid_mask]

        print("  Raw data stats")
        stats = self._get_statistics_2D(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

        # Standardize signals based on training data
        if None in [self.signal_mean, self.signal_stdev]:
            assert dataset_stage == 'train' or not self.train_events, f"Dataset_stage: {dataset_stage}"
            print(f"  Calculating signal mean and std from {dataset_stage} data")
            self.signal_mean = stats['mean']
            self.signal_stdev = stats['stdev']
            self.signal_exkurt = stats['exkurt']
            self.save_hyperparameters({
                'signal_mean': self.signal_mean.item(),
                'signal_stdev': self.signal_stdev.item(),
                'signal_exkurt': self.signal_exkurt.item(),
            })

        if dataset_stage in ['train', 'validation', 'test']:
            print(f"  Standardizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f}")
            packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev  # Standardize all signals
            stats = self._get_statistics_2D(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )

        self.max_abs_valid_signal = np.max(np.abs([stats['min'], stats['max']]))


        if dataset_stage in ['train']:
            dataset = TrainValTest_Dataset_3(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                signal_window_size=self.signal_window_size,
                time_points=packaged_times,
                radial_positions=packaged_radial_positions
            )
            return dataset
        if dataset_stage in ['validation', 'test']:
            self.datasets[dataset_stage] = TrainValTest_Dataset_3(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                signal_window_size=self.signal_window_size,
                time_points=packaged_times,
                radial_positions=packaged_radial_positions
            )
        
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _load_and_preprocess_predict_data(self, shot_start_time_indices):
        """
        Prepares the dataset for prediction, ensuring signals and labels are standardized
        using training stats, and labels are clipped as needed.
        """
        print("Preparing data for prediction with preprocessing")

        # List for storing (signal_window, label, time_point) tuples
        data = []
        print(f"Shots being processed: {set(shot for shot, _ in shot_start_time_indices)}")

        with h5py.File(self.data_file, 'r') as h5_file:
            for shot, event in shot_start_time_indices:
                print(f"shot: {shot}")
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Extract signals, times, and labels
                signals = np.array(event_data["signals"], dtype=np.float32)  # Shape: (64, time)
                times = np.array(event_data["times"], dtype=np.float32)  # Shape: (time,)
                label_times = np.array(event_data["label_times"][:], dtype=np.float32)  # shape (label_times,)
                target = np.array(event_data["vZ"][:, :], dtype=np.float32)  # shape (label_times, 8)

                # Skip if the signal length is insufficient
                if signals.shape[1] < self.signal_window_size:
                    print(f"Skipping event {event_key}: signal length ({signals.shape[1]}) < window size ({self.signal_window_size})")
                    continue

                # Map signal times to label indices
                label_indices = np.searchsorted(label_times, times, side='left')
                label_indices[label_indices >= len(label_times)] = len(label_times) - 1
                mask = (label_indices > 0) & (
                    np.abs(times - label_times[label_indices - 1]) < np.abs(times - label_times[label_indices])
                )
                label_indices[mask] -= 1

                # Align labels with signal times
                labels_aligned = target[label_indices, :self.n_cols]  # shape (time, n_cols)

                # Clip labels if required
                if self.clip_labels:
                    valid_mask = (labels_aligned >= self.labels_lower_bound) & (labels_aligned <= self.labels_upper_bound)
                    valid_mask = np.all(valid_mask, axis=1)  # Combine mask across columns
                    signals = signals[:, valid_mask]
                    labels = labels_aligned[valid_mask]
                    times = times[valid_mask]

                # Standardize or normalize labels
                if self.standardize_labels:
                    labels = (labels - self.label_mean) / self.label_std
                elif self.normalize_labels:
                    labels = -1 + 2 * (labels - self.label_min) / (self.label_max - self.label_min)

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Reshape signals to (time, n_rows, n_cols)
                signals = self.reshape_signals(signals, inboard_order)
                mean = np.mean(signals, axis=0, keepdims=True)
                std = np.std(signals, axis=0, keepdims=True)
                
                # Avoid division by zero by setting std to 1 where it is zero
                std[std == 0] = 1
                
                # Standardize signals
                signals = (signals - mean) / std
                
                # Standardize signals using training stats
                # signals = (signals - self.signal_mean) / self.signal_stdev

                # Ensure dimensions match
                assert signals.shape[0] == labels.shape[0] == times.shape[0], \
                    f"Dimension mismatch: signals={signals.shape[0]}, labels={labels.shape[0]}, times={times.shape[0]}"

                # Generate signal-window tuples for the dataset
                max_index = signals.shape[0] - self.signal_window_size + 1
                if max_index <= 0:
                    print(f"Skipping event {event_key}: not enough data for the specified window size")
                    continue

                for i in range(max_index):
                    signal_window = signals[i:i + self.signal_window_size]
                    label = labels_aligned[i + self.signal_window_size - 1]
                    time_point = times[i + self.signal_window_size - 1]
                    data.append((signal_window, label, time_point, shot))  # Include shot ID

        # Create and return a dataset for prediction
        return PredictDataset(data)

    def _load_and_preprocess_predict_data_4(self, shot_start_time_indices):
        """
        Prepares the dataset for prediction based on the new data structure, 
        including radial_positions for each sample.
        """
        print("Preparing data for prediction with radial positions")

        data = []  # Will store tuples (signal_window, label, time_point, shot_id, radial_position)
        unique_shots = set(shot for shot, _ in shot_start_time_indices)
        print(f"Shots being processed: {unique_shots}")

        with h5py.File(self.data_file, 'r') as h5_file:
            # Filter shot_start_time_indices for valid events
            valid_events = []
            for shot, event in shot_start_time_indices:
                event_key = f"{shot}/{event}"
                if event in h5_file[shot].keys():
                    if 'signals' in h5_file[event_key] and 'times' in h5_file[event_key]:
                        valid_events.append((shot, event))
                    else:
                        print(f"Skipping {event_key} - missing signals or times.")
                else:
                    print(f"Skipping {event_key} - event not found.")

            for i, (shot, start_time) in enumerate(valid_events):
                if i % 100 == 0:
                    print(f"  Reading prediction event {i:04d}/{len(valid_events):04d}, start_time: {start_time} in shot {shot}")

                event_key = f"{shot}/{start_time}"
                event_data = h5_file[event_key]

                signals = np.array(event_data["signals"], dtype=np.float32)  # shape (64, time)
                times = np.array(event_data["times"], dtype=np.float32)      # shape (time,)

                if signals.shape[1] < self.signal_window_size:
                    print(f"Skipping event {event_key}: signal length < window size ({self.signal_window_size})")
                    continue

                # Load shot-level data
                shot_group = h5_file[shot]
                label_times = np.array(shot_group["label_times"], dtype=np.float32)  # shape (label_times,)
                target = np.array(shot_group[self.target_labels[0]], dtype=np.float32)       # shape (label_times, n_cols)
                uncertainty = np.array(shot_group[self.target_labels[1]], dtype=np.float32)  # shape (label_times, n_cols)

                inboard_order = shot_group.attrs.get("inboard_column_channel_order", None)
                if inboard_order is None or len(inboard_order) == 0:
                    print(f"Skipping event {event_key}: Missing or empty inboard_column_channel_order.")
                    continue

                signals = self.reshape_signals(signals, inboard_order)  # (time, n_rows, n_cols)

                # Apply optional bandpass filter
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = self.apply_bandpass_filter(signals)
                    signals = signals.astype(np.float32)

                r_position = shot_group.attrs.get("r_position", None)
                if r_position is not None and len(r_position) == 64:
                    radial_positions = r_position.reshape(8, 8)[0, :].astype(np.float32)
                else:
                    print(f"Warning: Radial positions missing or malformed for shot {shot}.")
                    radial_positions = np.zeros(self.n_cols, dtype=np.float32)

                labels_aligned = self._filter_and_transform_predict_labels(label_times, target, times)

                # Skip if all labels are NaN
                if np.isnan(labels_aligned).all():
                    continue
                
                # Remove rows where any column label is NaN
                valid_mask = ~np.isnan(labels_aligned).any(axis=1)
                signals = signals[valid_mask, :, :]
                labels_aligned = labels_aligned[valid_mask, :]
                times = times[valid_mask]

                # Standardize signals based on training stats
                signals = (signals - self.signal_mean) / self.signal_stdev

                total_time = signals.shape[0]
                max_index = total_time - self.signal_window_size + 1
                if max_index <= 0:
                    print(f"Skipping event {event_key}: not enough data for the specified window size")
                    continue

                # For each column, treat it like a separate feature set
                for col in range(self.n_cols):
                    col_signals = signals[:, :, col]  # shape (time, n_rows)
                    col_labels = labels_aligned[:, col]
                    col_radial_position = radial_positions[col]

                    if col_signals.size == 0 or col_labels.size == 0:
                        continue

                    # Generate samples for each valid window index
                    for idx_t0 in range(self.signal_window_size - 1, total_time):
                        start_idx = idx_t0 - self.signal_window_size + 1
                        end_idx = idx_t0 + 1
                        signal_window = col_signals[start_idx:end_idx, :]  # (window_size, n_rows)
                        label = col_labels[idx_t0]
                        time_point = times[idx_t0]

                        # Store sample including radial_position
                        data.append((signal_window, label, time_point, shot, col_radial_position))

        print(f"Prepared {len(data)} samples for prediction.")
        return PredictDataset_4(data)

    def _filter_and_transform_labels(self, label_times, target, uncertainty, times):
        """
        Filter labels based on uncertainty, align with signal times, clip to bounds, 
        and optionally standardize or normalize the labels.
        """
        # Step 1: Filter labels based on uncertainty threshold
        valid_mask = uncertainty <= self.vZ_uncertainty_threshold
        target[~valid_mask] = np.nan  # Set invalid values to NaN

        # Step 2: Align labels to signal times
        label_indices = np.searchsorted(label_times, times, side='left')
        label_indices[label_indices >= len(label_times)] = len(label_times) - 1
        mask = (label_indices > 0) & (
            np.abs(times - label_times[label_indices - 1]) < np.abs(times - label_times[label_indices])
        )
        label_indices[mask] -= 1
        labels_aligned = target[label_indices, :self.n_cols]  # shape (time, n_cols)

        # Step 3: Clip labels to the specified range (if enabled)
        if self.clip_labels:
            # print(f"    Clipping labels to range [{self.labels_lower_bound}, {self.labels_upper_bound}]...")
            labels_aligned = np.clip(labels_aligned, self.labels_lower_bound, self.labels_upper_bound)

        # Step 4: Apply label transformations (standardization or normalization)
        if self.standardize_labels:
            # print(f"    Standardizing labels...")
            labels_aligned = (labels_aligned - self.label_mean) / self.label_std

        elif self.normalize_labels:
            # print(f"    Normalizing labels to range [-1, 1]...")
            labels_aligned = -1 + 2 * (labels_aligned - self.label_min) / (self.label_max - self.label_min)

        labels_aligned = labels_aligned.astype(np.float32)
        return labels_aligned

    def _filter_and_transform_predict_labels(self, label_times, target, times):
        """
        Filter labels based on uncertainty, align with signal times, clip to bounds, 
        and optionally standardize or normalize the labels.
        """

        # Step 2: Align labels to signal times
        label_indices = np.searchsorted(label_times, times, side='left')
        label_indices[label_indices >= len(label_times)] = len(label_times) - 1
        mask = (label_indices > 0) & (
            np.abs(times - label_times[label_indices - 1]) < np.abs(times - label_times[label_indices])
        )
        label_indices[mask] -= 1
        labels_aligned = target[label_indices, :self.n_cols]  # shape (time, n_cols)

        labels_aligned = labels_aligned.astype(np.float32)
        return labels_aligned

    def reshape_signals_8x8(self, signals, inboard_order):
        # Assumptions:
        # - `inboard_order` contains valid indices for the starting positions of each row.
        # - `signals` is expected to be of shape (num_channels, num_samples), where num_channels >= max(inboard_order) + 7.

        # truncate the inboard_order array to first 8 rows
        inboard_order = inboard_order[:8]

        # Initialize the reshaped signals with zeros or np.nan if there's a chance of not filling some cells
        reshaped_signals = np.zeros((signals.shape[1], 8, 8), dtype=np.float32)  # Using zeros as default values

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
    
    def reshape_signals(self, signals, inboard_order):
        # Assumptions:
        # - `inboard_order` contains valid indices for the starting positions of each row.
        # - `signals` is expected to be of shape (num_channels, num_samples), where num_channels >= max(inboard_order) + 7.

        # truncate the inboard_order array to first 8 rows
        inboard_order = inboard_order[:self.n_rows]

        # Initialize the reshaped signals with zeros or np.nan if there's a chance of not filling some cells
        reshaped_signals = np.zeros((signals.shape[1], self.n_rows, self.n_cols), dtype=np.float32)  # Using zeros as default values

        # Reshape signals according to the truncated inboard_order
        for row, start_idx in enumerate(inboard_order):
            for col in range(self.n_cols):
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

            if signals.shape[0] > required_length:
                # Design the bandpass filter
                bandpass_filter = firwin(
                    self.filter_taps,
                    [self.lower_cutoff_frequency_hz, self.upper_cutoff_frequency_hz],
                    pass_zero=False,
                    fs=self.sampling_frequency_hz
                )

                # Apply the filter
                filtered_signals = filtfilt(bandpass_filter, 1, signals, axis=0)
                filtered_signals = filtered_signals.astype(np.float32)
                return filtered_signals
            else:
                # print("BANDPASS FILTER NOT APPLIED")
                return signals

    def apply_highpass_filter(self, signals):
        """
        Applies a high-pass filter to the given signals if the cutoff frequency is specified.
        Otherwise, returns the original signals.

        Args:
            signals: The signals to be filtered.

        Returns:
            Filtered signals or the original signals.
        """
        required_length = 3 * self.filter_taps  # Set to 3 times the number of filter taps

        # Check if the cutoff frequency is specified
        if signals.shape[0] > required_length:
            # Design the high-pass filter
            highpass_filter = firwin(
                self.filter_taps,
                self.highpass_cutoff_frequency_hz,
                pass_zero="highpass",
                fs=1e5, # sampling rate after downsampling by factor of 10
            )

            # Apply the filter
            filtered_signals = filtfilt(highpass_filter, 1, signals, axis=0)
            return filtered_signals
        else:
            # print("HIGH-PASS FILTER NOT APPLIED")
            return signals
        
    def _get_events_and_split(self):
        """
        Load all events once, store them, and then pick a splitting strategy
        depending on self.split_method.
        """
        events = self._load_events()
        self.all_events = events  # Store in case we need them for debugging

        if self.split_method == "shot":
            self._assign_datasets_shot(events)
        elif self.split_method == "event":
            self._assign_datasets_event(events)
        else:
            raise ValueError(f"Invalid split_method='{self.split_method}'. "
                            "Must be 'shot' or 'event'.")    
                   
    # def _get_events_and_split(self):
    #     events = self._load_events()
    #     self._assign_datasets(events)

    def _assign_datasets_event(self, events):
        """
        New approach: Shuffle all events (ignoring shot boundaries) and split
        into train/val/test according to fraction_validation/fraction_test.
        """
        # 1) If we have specific shots for 'predict', separate them first
        predict_events = []
        if self.predict_shots:
            predict_shot_set = set(self.predict_shots)
            predict_events = [e for e in events if e[0] in predict_shot_set]
            # Only keep the events that are NOT in predict_shot_set for normal splitting
            # events = [e for e in events if e[0] not in predict_shot_set]

        # 2) Shuffle all remaining events
        np.random.seed(self.seed)
        np.random.shuffle(events)

        n_events = len(events)

        # 3) If you prefer direct fraction-of-all: 
        #    e.g. 20% val, 20% test => 60% train
        n_val = int(self.fraction_validation * n_events)
        n_test = int(self.fraction_test * n_events)

        validation_events = events[:n_val]
        test_events = events[n_val:n_val + n_test]
        train_events = events[n_val + n_test:]

        # 4) Assign
        self.train_events = train_events
        self.validation_events = validation_events
        self.test_events = test_events
        self.predict_events = predict_events

        # 5) Print info
        print(f"Train set size: {len(self.train_events)} events")
        print(f"Validation set size: {len(self.validation_events)} events")
        print(f"Test set size: {len(self.test_events)} events")
        print(f"Predict set size: {len(self.predict_events)} events")

    def _assign_datasets_shot(self, events):
        # Extract all available shots in the data file
        available_shots = set(shot for shot, _ in events)

        # Assign shots for each dataset
        train_shots = self.train_shots if self.train_shots is not None else []
        validation_shots = self.validation_shots if self.validation_shots is not None else []
        test_shots = self.test_shots if self.test_shots is not None else []
        predict_shots = self.predict_shots if self.predict_shots is not None else []

        # Check for missing shots
        missing_train_shots = [shot for shot in train_shots if shot not in available_shots]
        missing_validation_shots = [shot for shot in validation_shots if shot not in available_shots]
        missing_test_shots = [shot for shot in test_shots if shot not in available_shots]
        missing_predict_shots = [shot for shot in predict_shots if shot not in available_shots]

        # Warn if shots are missing
        if missing_train_shots:
            print(f"Warning: The following train shots are missing from the data file: {missing_train_shots}")
        if missing_validation_shots:
            print(f"Warning: The following validation shots are missing from the data file: {missing_validation_shots}")
        if missing_test_shots:
            print(f"Warning: The following test shots are missing from the data file: {missing_test_shots}")
        if missing_predict_shots:
            print(f"Warning: The following predict shots are missing from the data file: {missing_predict_shots}")

        # Filter events for each dataset
        train_events = [event for event in events if event[0] in train_shots]
        validation_events = [event for event in events if event[0] in validation_shots]
        test_events = [event for event in events if event[0] in test_shots]
        predict_events = [event for event in events if event[0] in predict_shots]

        # Special case: if both fraction_validation and fraction_test are 0, populate only the training set
        if self.fraction_validation == 0 and self.fraction_test == 0:
            self.train_events = train_events
            self.validation_events = []
            self.test_events = []
            print(f"Train set size: {len(self.train_events)} events (validation and test sets skipped)")
            return

        # If no validation shots are provided, take a portion from the training set
        if not validation_shots:
            np.random.seed(self.seed)
            train_events, validation_events = train_test_split(train_events, test_size=self.fraction_validation, random_state=self.seed)

        self.train_events = train_events
        self.validation_events = validation_events
        self.test_events = test_events
        self.predict_events = predict_events

        print(f"Train set size: {len(self.train_events)} events")
        print(f"Validation set size: {len(self.validation_events)} events")
        print(f"Test set size: {len(self.test_events)} events")
        print(f"Predict set size: {len(self.predict_events)} events")

    def _load_events(self):
        events = []
        with h5py.File(self.data_file, "r") as data_file:
            for shot in data_file.keys():
                shot_group = data_file[shot]
                for event in shot_group.keys():
                    # Check if this is an actual event group with signals/times
                    # First, ensure it's a group, not a dataset
                    if isinstance(shot_group[event], h5py.Group):
                        # Optionally check if 'signals' and 'times' are present
                        if 'signals' in shot_group[event] and 'times' in shot_group[event]:
                            events.append((shot, event))
        print(f"Loaded {len(events)} events from {len(set(shot for shot, _ in events))} unique shots.")
        return events

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
                
    def _get_valid_indices(self, labels: np.ndarray) -> np.ndarray:
        label_length = labels.shape[0]
        valid_t0 = np.zeros(label_length, dtype=int)
        first_valid_signal_window_start_index = self.signal_window_size - 1

        # Check for finite labels
        valid_labels = np.isfinite(labels).all(axis=1)  # Assuming labels have shape (time, ...)
        valid_t0[first_valid_signal_window_start_index:] = valid_labels[first_valid_signal_window_start_index:]

        return valid_t0

    def _get_valid_indices_3(self, labels: np.ndarray) -> np.ndarray:
        """
        Generate valid indices for a single column of labels, considering the flattened (time * n_cols) structure.
        Expects `labels` to have shape (time,).
        """
        label_length = labels.shape[0]
        valid_t0 = np.zeros(label_length, dtype=int)
        first_valid_signal_window_start_index = self.signal_window_size - 1

        # Check for finite labels
        valid_labels = np.isfinite(labels)  # Shape: (time,)
        valid_t0[first_valid_signal_window_start_index:] = valid_labels[first_valid_signal_window_start_index:]

        # Expand to match flattened structure (time * n_cols)
        expanded_valid_t0 = np.repeat(valid_t0, self.n_cols)

        return expanded_valid_t0

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
            if signal_window.size > 0:  # Check if the window is non-empty
                signal_min = min(signal_min, signal_window.min())
                signal_max = max(signal_max, signal_window.max())
                hist, bin_edges = np.histogram(
                    signal_window,
                    bins=n_bins,
                    range=[-10.4, 10.4],
                )
                cummulative_hist += hist
            else:
                continue  # Skip processing if the signal window is empty

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

    def _get_statistics_2D(
            self,
            sample_indices: np.ndarray,
            signals: np.ndarray,
    ) -> dict:
        """
        Calculate statistics for the given signal windows using sample_indices.
        Signals are 2D: (time * n_cols, n_rows).
        """
        signal_min = np.inf
        signal_max = -np.inf
        n_bins = 200
        cumulative_hist = np.zeros(n_bins, dtype=int)
        stat_samples = int(100e3)
        stat_interval = max(1, sample_indices.size // stat_samples)
        n_samples = sample_indices.size // stat_interval

        for i in sample_indices[::stat_interval]:
            start_index = max(0, i - self.signal_window_size + 1)  # Ensure valid start index
            signal_window = signals[start_index:i + 1, :]  # Shape: (window_size, n_rows)
            if signal_window.size > 0:  # Ensure the window is non-empty
                signal_min = min(signal_min, signal_window.min())
                signal_max = max(signal_max, signal_window.max())
                hist, bin_edges = np.histogram(
                    signal_window.flatten(),  # Flatten for histogram calculation
                    bins=n_bins,
                    range=[-10.4, 10.4],
                )
                cumulative_hist += hist

        bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        mean = np.sum(cumulative_hist * bin_center) / np.sum(cumulative_hist)
        stdev = np.sqrt(np.sum(cumulative_hist * (bin_center - mean) ** 2) / np.sum(cumulative_hist))
        exkurt = np.sum(cumulative_hist * ((bin_center - mean) / stdev) ** 4) / np.sum(cumulative_hist) - 3
        print(f"    Stats: count {sample_indices.size:,} min {signal_min:.3f} max {signal_max:.3f} mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} n_samples {n_samples:,}")
        return {
            'count': sample_indices.size,
            'min': signal_min,
            'max': signal_max,
            'mean': mean,
            'stdev': stdev,
            'exkurt': exkurt,
        }
    
    def custom_collate_fn(self, batch):
        # Extract all components: signals, labels, and possibly others
        signals = torch.stack([item[0] for item in batch])
        labels = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1].keys()}
        time_points = torch.stack([item[2] for item in batch])

        return signals, labels, time_points

    def train_dataloader(self):
        if self.split_train_data_per_gpu:
            return self._train_dataloader
        else:
            train_sampler = torch.utils.data.DistributedSampler(
                self.datasets['train'],
                shuffle=False,
                drop_last=True,
            )
            return torch.utils.data.DataLoader(
                dataset=self.datasets['train'],
                sampler=train_sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                # pin_memory=True,
                persistent_workers=(self.num_workers > 0),
            ) 
    
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
        if torch.distributed.is_initialized():
            # Use a DistributedSampler if distributed training is active
            predict_sampler = torch.utils.data.DistributedSampler(
                self.datasets['predict'],
                shuffle=False,
                drop_last=True,
            )
        else:
            # Otherwise, no sampler is needed
            predict_sampler = None

        return torch.utils.data.DataLoader(
            dataset=self.datasets['predict'],
            sampler=predict_sampler,
            batch_size=self.batch_size,
            shuffle=False,  # Shuffle is False since prediction typically follows a fixed order
            num_workers=4 if self.num_workers > 0 else 0,  # Adjust based on the environment
            pin_memory=True,  # Enable pin_memory for faster data transfers to GPU
            persistent_workers=(self.num_workers > 0),  # Keep workers alive if num_workers > 0
        )