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

class TrainValTest_Dataset_4(torch.utils.data.Dataset):
    """
    Windows over time of a selected BES sub-grid.

    Returns per __getitem__:
      - signal_window: (1, W, R_sel, C_sel)  # channel-first 3D block over time
      - label_scalar : () float32            # vθ(ψ_target) at window end time
      - time_point_ms: () float32            # t0 (ms)
    """
    def __init__(
        self,
        signals_trc: np.ndarray,      # (T_total, R_sel, C_sel), standardized if requested
        labels_scalar: np.ndarray,    # (T_total,)
        sample_indices: np.ndarray,   # (N_samples,), each is t0 index
        signal_window_size: int,      # W
        times_ms: np.ndarray,         # (T_total,)
        n_rows_sel: int,              # for sanity
        n_cols_sel: int,              # for sanity
    ):
        self.signals = torch.from_numpy(np.ascontiguousarray(signals_trc)).float()
        self.labels  = torch.from_numpy(labels_scalar.astype(np.float32))
        self.sample_indices = torch.from_numpy(sample_indices.astype(np.int64))
        self.times_ms = torch.from_numpy(times_ms.astype(np.float32))
        self.W = int(signal_window_size)
        T, R, C = self.signals.shape
        assert R == n_rows_sel and C == n_cols_sel, "Selected grid shape mismatch"
        assert self.labels.shape[0] == T == self.times_ms.shape[0], "T mismatch"
        assert self.sample_indices.numel() > 0, "Empty dataset"

    def __len__(self) -> int:
        return self.sample_indices.numel()

    def __getitem__(self, idx: int):
        i_t0 = int(self.sample_indices[idx].item())
        s = i_t0 - self.W + 1
        e = i_t0 + 1
        if s < 0 or e > self.signals.shape[0]:
            raise IndexError(f"Invalid window [{s}:{e}] for T={self.signals.shape[0]}")

        # (W, R, C) -> (1, W, R, C)
        win = self.signals[s:e, :, :].unsqueeze(0)  # add channel dim
        y   = self.labels[i_t0]
        t0  = self.times_ms[i_t0]
        return win, y, t0

    
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
            
class PredictDataset_5(torch.utils.data.Dataset):
    """
    Yields:
      signals : (1, W, R_sel, C_sel)  float32
      label   : scalar float32 (NaN if label unavailable)
      time_ms : scalar float32 (window end time)
      shot_id : Python str/int (kept as-is)
      event_id: Python str/int (kept as-is)
    """
    def __init__(self, windows, labels, times_ms, shots, events):
        self.windows  = windows
        self.labels   = labels
        self.times_ms = times_ms
        self.shots    = shots
        self.events   = events

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        x  = torch.from_numpy(self.windows[idx])   # (1,W,R_sel,C_sel)
        y  = torch.tensor(self.labels[idx], dtype=torch.float32)
        tm = torch.tensor(self.times_ms[idx], dtype=torch.float32)
        sh = self.shots[idx]
        ev = self.events[idx]
        return x, y, tm, sh, ev

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
    train_shots: list = None
    validation_shots: list = None
    test_shots: list = None
    predict_shots: list = None
    # Bandpass filter parameters
    sampling_frequency_hz: float = 1 / 10**(-6)  # Sampling frequency in Hz
    downsample_factor: float = 1
    filter_taps: int = 501  # Number of taps in the filter
    lower_cutoff_frequency_hz: float = None  # Lower cutoff frequency in Hz
    upper_cutoff_frequency_hz: float = None  # Upper cutoff frequency in Hz
    highpass_cutoff_frequency_hz: float = None # Lower cutoff frequency in Hz (typically used after downsampling BES signals)
    standardize_signals: bool = True
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
    standardize_labels: bool = False  # Option to standardize labels
    normalize_labels: bool = False   # Option to normalize labels
    label_mean: float = 0.0           # mean value for standardization
    label_std: float = 1.0            # std dev value for standardization
    label_min: float = -2e9           # min value for normalization
    label_max: float = 2e9            # max value for normalization
    vZ_uncertainty_threshold: float = 200.0
    do_flip_augmentation: bool = False
    predict_window_stride: int = 1  
    split_train_data_per_gpu: bool = True  
    num_train_batches_per_gpu: int = None
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    is_global_zero: bool = dataclasses.field(default=True, init=False)
    log_dir: str = dataclasses.field(default='.', init=False)
    world_size: int = 1 # number of total GPUs 
    shot_radial_time_windows: dict = None
    shot_time_windows: dict = None
    train_time_windows: dict = None   # dict like shot_time_windows (optional)
    # ---- NEW: block selection & labels-at-fixed-psi ----
    block_cols: typing.Any = ('last', 4)   # ('last', k) | list[int] | slice
    row_stride: int = 1                     # 1 = keep every row
    row_offset: int = 0                     # starting row offset (0..row_stride-1)
    target_sampling_hz: float = 250_000.0   # downsample target (Hz)
    label_target_psi: float = 0.90          # e.g. 0.90 -> "psi_0p9"
    label_tolerance_ms: float = 0.6         # nearest-neighbor tolerance
    window_hop: int = 1                     # stride between successive t0's (in samples at target rate)
    # ----------------------------------------------------

    def __post_init__(self):
        super().__init__()
        if self.data_file is None:
            self.data_file = sample_elm_data_file.as_posix()
        self.save_hyperparameters(
            ignore=['max_predict_elms', 'n_rows', 'n_cols']
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
                    dataset = self._load_and_preprocess_data_6(chunk_events, dataset_stage)

                    if dataset_stage == "train":
                        local_batches = len(dataset) // self.batch_size
                        print(f"[rank {global_rank}] local train batches: {local_batches}")

                        # Compute global min across ranks (NCCL -> must be CUDA tensor)
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            t = torch.tensor(local_batches, device="cuda", dtype=torch.long)
                            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                            min_batches = int(t.item())
                        else:
                            min_batches = local_batches

                        if min_batches < 1:
                            raise RuntimeError(
                                f"min_batches={min_batches}. At least one rank has <1 full batch. "
                                f"Reduce world_size, lower batch_size, or change your chunking."
                            )

                        self.num_train_batches_per_gpu = min_batches
                        min_samples = min_batches * self.batch_size

                        # Truncate so EVERY rank has the same number of batches
                        dataset = torch.utils.data.Subset(dataset, range(min_samples))
                        print(f"[rank {global_rank}] trunc to {min_samples} samples = {min_batches} batches")

                        self._train_dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            persistent_workers=(self.num_workers > 0),
                            drop_last=True,
                        )

                elif dataset_stage in ['predict']:
                    print(f"Preparing predict dataset for shots: {self.predict_shots}")
                    # Load and preprocess the data for these events
                    dataset = self._load_and_preprocess_predict_data_6(chunk_events)
                    # Assign the predict dataset
                    self.datasets["predict"] = dataset

        else:
            for dataset_stage in dataset_stages:
                events = self.dataset_events[dataset_stage]

                if dataset_stage in ['train', 'validation', 'test']:
                    dataset = self._load_and_preprocess_data_6(events, dataset_stage)
                elif dataset_stage in ['predict']:
                    # Load and preprocess the data for these events
                    dataset = self._load_and_preprocess_predict_data_6(chunk_events)
                    # Assign the predict dataset
                    self.datasets["predict"] = dataset

    def _cols_from_spec(self, spec, n=8):
        import numpy as np
        if isinstance(spec, tuple) and spec[0] == 'last':
            k = int(spec[1]); return np.arange(n-k, n)
        if isinstance(spec, slice):
            return np.arange(n)[spec]
        if isinstance(spec, (list, np.ndarray)):
            return np.array(spec, dtype=int)
        raise ValueError("bad block_cols")
    
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
        
    def _load_and_preprocess_data_5(self, shot_start_time_indices, dataset_stage):
        """
        Load and preprocess BES data for the machine learning pipeline.
        Processes signals column-wise, applies masking for high uncertainty, and integrates radial positions.
        Now the vZ, vZ_uncertainty, label_times, etc. are saved in an "interpolated" subgroup under each shot.
        """
        import time  # in case not already imported
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
                signals = np.array(event_data["signals"], dtype=np.float32)  # shape: (64, time)
                times = np.array(event_data["times"], dtype=np.float32)      # shape: (time,)
                signal_length = signals.shape[1]
                if signal_length < self.signal_window_size:
                    continue

                # Apply time mask based on start_time_ms/end_time_ms
                time_mask = np.ones_like(times, dtype=bool)
                if self.start_time_ms is not None:
                    time_mask &= (times >= self.start_time_ms)
                if self.end_time_ms is not None:
                    time_mask &= (times <= self.end_time_ms)
                    if times.size < self.signal_window_size:
                        continue

                # Load shot-level data from the shot group.
                shot_group = h5_file[shot]
                # Instead of checking for "radial_0", we now simply check if an "interpolated" subgroup exists.
                if "interpolated" not in shot_group:
                    print(f"Skipping shot {shot}: no interpolated subgroup exists.")
                    continue

                # Reshape raw signals (assumed to be stored at the event level) to (time, n_rows, n_cols)
                inboard_order = shot_group.attrs["inboard_column_channel_order"]
                signals = self.reshape_signals(signals, inboard_order)  # New shape: (time, n_rows, n_cols)

                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter")
                    signals = self.apply_bandpass_filter(signals)

                # Retrieve radial positions
                r_position = shot_group.attrs["r_position"]
                if r_position is not None and len(r_position) == 64:
                    radial_positions = r_position.reshape(8, 8)[0, :]  # extract unique positions from first row
                else:
                    print(f"Warning: Radial positions missing or malformed for shot {shot}.")
                    radial_positions = np.zeros(self.n_cols)

                # Process each radial column (each "channel") separately.
                for col in range(self.n_cols):
                    col_signals = signals[:, :, col]  # shape: (time, rows)
                    # ***********************
                    # NEW: Instead of getting the labels from the top-level, load the radial-specific data 
                    # from the "interpolated/radial_{col}" subgroup.
                    # ***********************
                    if "interpolated" in shot_group and f"radial_{col}" in shot_group["interpolated"]:
                        interp_grp = shot_group["interpolated/radial_" + str(col)]
                        col_label_times = np.array(interp_grp["label_times"], dtype=np.float32)  # in ms
                        col_target = np.array(interp_grp["vZ"], dtype=np.float32)                # corresponding vZ data
                        col_uncertainty = np.array(interp_grp["vZ_uncertainty"], dtype=np.float32)  # corresponding uncertainty
                    else:
                        print(f"Skipping column {col} in shot {shot}: interpolated data missing for this channel.")
                        continue
                    # ***********************

                    # For alignment, decide whether to align event-level times or use the interpolated grid.
                    # Here we assume that the event-level times (possibly after conversion) are used.
                    # (If event times are in µs and label_times in ms, you may need to divide times by 1000.)
                    col_labels = self._filter_and_transform_labels_5(col_label_times, col_target, col_uncertainty, times)

                    # --- New: Skip channel if all labels are NaN ---
                    if np.isnan(col_labels).all():
                        continue

                    # --- New: Remove NaN labels along the time axis ---
                    valid_time_mask = ~np.isnan(col_labels)  # Now valid_time_mask has shape (T,)
                    # Apply the valid mask to both the signals and the event times:
                    col_signals = col_signals[valid_time_mask, :]   # Now col_signals has shape (~T, rows)
                    col_labels   = col_labels[valid_time_mask]        # Shape: (~T,)
                    times_event_filtered = times[valid_time_mask]     # Shape: (~T,)

                    # If shot_radial_time_windows exist for this shot & column, then further restrict the samples.
                    if (hasattr(self, "shot_radial_time_windows") and 
                        self.shot_radial_time_windows is not None and 
                        shot in self.shot_radial_time_windows and 
                        col in self.shot_radial_time_windows[shot]):
                        
                        windows = self.shot_radial_time_windows[shot][col]
                        specific_time_mask = np.zeros_like(times_event_filtered, dtype=bool)
                        for window in windows:
                            window_mask = (times_event_filtered >= window[0]) & (times_event_filtered <= window[1])
                            specific_time_mask |= window_mask
                        
                        if np.sum(specific_time_mask) < self.signal_window_size:
                            continue

                        # Further restrict col_labels and the time array.
                        col_labels = col_labels[specific_time_mask]
                        col_signals = col_signals[specific_time_mask, :]
                        col_times = times_event_filtered[specific_time_mask]
                        valid_t0 = np.zeros(col_times.shape[0], dtype=int)
                        valid_t0[self.signal_window_size - 1::self.signal_window_size] = 1

                    else:
                        # If no specific window for this radial column, skip.
                        continue

                    signals_list.append(col_signals)
                    labels_list.append(col_labels)
                    valid_t0_list.append(valid_t0)
                    radial_positions_list.append(np.full(col_labels.shape, radial_positions[col]))
                    times_list.append(col_times)

                    # Optional: add flipped augmentation if enabled
                    if self.do_flip_augmentation:
                        col_signals_flipped = col_signals[:, ::-1]  # Flip rows
                        col_labels_flipped = -col_labels             # Reverse sign
                        signals_list.append(col_signals_flipped)
                        labels_list.append(col_labels_flipped)
                        valid_t0_list.append(valid_t0)
                        radial_positions_list.append(np.full(col_labels.shape, radial_positions[col]))
                        times_list.append(col_times)

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

    def _load_and_preprocess_data_6(self, shot_start_time_indices, dataset_stage: str):
        """
        New: builds block inputs (rows x cols) and scalar labels at fixed ψ.
        Windows are (1, signal_window_size, R_sel, C_sel); label is vθ(ψ) at window end time.
        """

        import numpy as np, h5py, torch
        from scipy.signal import decimate, resample_poly
        from fractions import Fraction
        import time

        t_start = time.time()
        print(f"[{dataset_stage}] load+preprocess (ψ={self.label_target_psi}, target_fs={self.target_sampling_hz} Hz)")

        # -------- helpers (scoped) --------
        def _format_psi_key(psi: float) -> str:
            s = f"{psi:.3f}".rstrip('0').rstrip('.')
            return "psi_" + s.replace('.', 'p')

        def _guess_times_ms(times: np.ndarray) -> np.ndarray:
            if times.size < 2: return times.astype(np.float32)
            dt = np.median(np.diff(times))
            # 1 MHz sampling is often stored in microseconds (dt≈1); convert to ms.
            if 0.5 <= dt <= 2.0 and times.max() > 5e3:
                return (times / 1000.0).astype(np.float32)
            return times.astype(np.float32)

        def _downsample_1d(arr, orig_fs, target_fs, axis=0):
            if orig_fs == target_fs: return arr
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                return decimate(arr, q, ftype='fir', axis=axis, zero_phase=True)
            frac = Fraction(target_fs / orig_fs).limit_denominator(64)
            return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

        def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
            x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
            new_len = x.shape[0]
            t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
            return x.astype(np.float32), t

        def _align_label_scalar_at_times(t_ms, label_t_ms, label_vals, tol_ms=0.6):
            if t_ms.size == 0 or label_t_ms.size == 0:
                return np.full_like(t_ms, np.nan, dtype=np.float32)
            idx = np.searchsorted(label_t_ms, t_ms)
            idx0 = np.clip(idx - 1, 0, label_t_ms.size - 1)
            idx1 = np.clip(idx,     0, label_t_ms.size - 1)
            d0 = np.abs(t_ms - label_t_ms[idx0])
            d1 = np.abs(t_ms - label_t_ms[idx1])
            choose0 = d0 <= d1
            chosen_idx = np.where(choose0, idx0, idx1)
            chosen_dist = np.where(choose0, d0, d1)
            out = np.full(t_ms.shape, np.nan, dtype=np.float32)
            ok = chosen_dist <= tol_ms
            out[ok] = label_vals[chosen_idx[ok]].astype(np.float32)
            return out

        def _window_indices(valid_mask: np.ndarray, W: int, hop: int = 1) -> np.ndarray:
            T = valid_mask.size
            cs = np.cumsum(valid_mask.astype(np.int32))
            def wsum(t0):
                start = t0 - W + 1
                if start <= 0: return cs[t0]
                return cs[t0] - cs[start-1]
            t0s = []
            for t0 in range(W-1, T, hop):
                if wsum(t0) == W:
                    t0s.append(t0)
            return np.array(t0s, dtype=np.int64)

        def _gather_shot_windows_mask(shot: str, t_ms: np.ndarray):
            # Support either: shot_time_windows = {shot: [(t0,t1), ...]}
            # or legacy shot_radial_time_windows = {shot: {col: [(t0,t1), ...], ...}}
            if getattr(self, "shot_time_windows", None) and shot in self.shot_time_windows:
                mask = np.zeros_like(t_ms, dtype=bool)
                for t0, t1 in self.shot_time_windows[shot]:
                    mask |= (t_ms >= t0) & (t_ms <= t1)
                return mask
            if getattr(self, "shot_radial_time_windows", None) and shot in self.shot_radial_time_windows:
                mask = np.zeros_like(t_ms, dtype=bool)
                for _col, wins in self.shot_radial_time_windows[shot].items():
                    for t0, t1 in wins:
                        mask |= (t_ms >= t0) & (t_ms <= t1)
                return mask
            return np.ones_like(t_ms, dtype=bool)
        
        def _mask_from_windows_dict(shot: str, t_ms: np.ndarray, windows_dict):
            # windows_dict: {shot: [(t0,t1), ...]} ; supports "all" or "*" fallback
            if not windows_dict:
                return np.ones_like(t_ms, dtype=bool)

            wins = windows_dict.get(shot, None)
            if wins is None:
                wins = windows_dict.get("all", windows_dict.get("*", None))
            if wins is None or len(wins) == 0:
                return np.ones_like(t_ms, dtype=bool)

            # allow shorthand "use all times"
            if len(wins) == 1 and (wins[0] == () or wins[0] == (None, None)):
                return np.ones_like(t_ms, dtype=bool)

            mask = np.zeros_like(t_ms, dtype=bool)
            for w in wins:
                if w == () or w is None:
                    return np.ones_like(t_ms, dtype=bool)
                t0, t1 = w
                if t0 is None: t0 = -np.inf
                if t1 is None: t1 =  np.inf
                mask |= (t_ms >= t0) & (t_ms <= t1)
            return mask

        def _get_stats_over_windows(sig_trc: np.ndarray, sample_idx: np.ndarray, W: int):
            # mean/std over (time within window × rows × cols), sampled over many windows
            T, R, C = sig_trc.shape
            idx = sample_idx
            if idx.size > 20000:
                idx = idx[np.linspace(0, idx.size-1, 20000, dtype=int)]
            acc_sum = 0.0; acc_sq = 0.0; count = 0
            for i_t0 in idx:
                s = int(i_t0 - W + 1); e = int(i_t0 + 1)
                if s < 0 or e > T: continue
                x = sig_trc[s:e]
                acc_sum += x.sum()
                acc_sq  += np.square(x).sum()
                count   += x.size
            mean = acc_sum / max(count, 1)
            var  = acc_sq / max(count, 1) - mean*mean
            return float(mean), float(max(var, 0.0) ** 0.5)

        # -------- selections --------
        W   = int(self.signal_window_size)
        hop = int(self.window_hop)
        orig_fs = float(self.sampling_frequency_hz)   # typically 1e6
        tgt_fs  = float(self.target_sampling_hz)
        psi_key = _format_psi_key(self.label_target_psi)

        # Build indices ON THE CANONICAL 8×8 GRID
        row_idx = np.arange(8)[self.row_offset::self.row_stride]          # e.g., every row or every other row, etc.
        col_idx = self._cols_from_spec(self.block_cols, n=8)               # e.g., ('last',4)->[4,5,6,7]
        R_sel, C_sel = row_idx.size, col_idx.size

        all_sig, all_lbl, all_t = [], [], []
        all_sample_idx = []
        base = 0
        used_events = 0
           
        with h5py.File(self.data_file, 'r') as h5:
            if len(shot_start_time_indices) >= 5:
                print("  head events:", shot_start_time_indices[:5])

            for shot, event in shot_start_time_indices:
                skey = f"{shot}/{event}"
                if skey not in h5:
                    print(f"  missing {skey}");  continue
                grp = h5[skey]
                if 'signals' not in grp or 'times' not in grp:
                    print(f"  {skey} missing 'signals' or 'times'");  continue

                shot_grp = h5[str(shot)]
                if 'inboard_column_channel_order' not in shot_grp.attrs:
                    print(f"  shot {shot}: missing inboard_column_channel_order");  continue

                # raw -> (T,8,8)
                sig_64t = np.array(grp['signals'], dtype=np.float32)        # (64, T)
                times   = np.array(grp['times'],   dtype=np.float64)        # (T,)
                order_attr = shot_grp.attrs.get('inboard_column_channel_order', None)
                sig_trc = self.reshape_signals_2(sig_64t, order_attr)  # -> (T, R, C)
                t_ms    = _guess_times_ms(times)

                # PAD to 8×8 so row_idx/col_idx are always valid
                sig_trc = self._pad_to_full_grid(sig_trc, target_R=8, target_C=8)  # (T, 8, 8)

                # sub-grid select
                sig_trc = sig_trc[:, row_idx[:, None], col_idx[None, :]]

                # optional bandpass (your function expects (T,R,C))
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    sig_trc = self.apply_bandpass_filter(sig_trc)

                # restrict by absolute time bounds (if set) BEFORE downsampling
                mask_time = np.ones_like(t_ms, dtype=bool)
                if self.start_time_ms is not None: mask_time &= (t_ms >= self.start_time_ms)
                if self.end_time_ms   is not None: mask_time &= (t_ms <= self.end_time_ms)
                sig_trc = sig_trc[mask_time];  t_ms = t_ms[mask_time]
                if sig_trc.shape[0] < W:  continue

                # downsample to target rate
                sig_ds, t_ds = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)

                # shot windows mask
                shot_mask = _gather_shot_windows_mask(str(shot), t_ds)

                # train-only extra restriction
                if dataset_stage == "train":
                    shot_mask &= _mask_from_windows_dict(str(shot), t_ds, getattr(self, "train_time_windows", None))

                # labels @ fixed ψ
                if "interpolated_psi" not in shot_grp or psi_key not in shot_grp["interpolated_psi"]:
                    print(f"  shot {shot}: no labels for {psi_key}");  continue
                psi_grp = shot_grp["interpolated_psi"][psi_key]
                lt  = np.array(psi_grp["label_times"], dtype=np.float32)
                lv  = np.array(psi_grp["vZ"],          dtype=np.float32)
                lbl = _align_label_scalar_at_times(t_ds, lt, lv, tol_ms=self.label_tolerance_ms)

                # final valid mask
                valid = shot_mask & ~np.isnan(lbl)
                # if valid.sum() < W:  continue
                sample_idx_ev = _window_indices(valid, W, hop=hop)
                if sample_idx_ev.size == 0:
                    continue

                # all_sig.append(sig_ds[valid])     # (Tv, R_sel, C_sel)
                # all_lbl.append(lbl[valid])        # (Tv,)
                # all_t.append(t_ds[valid])         # (Tv,)
                
                # used_events += 1

                all_sig.append(sig_ds)     # (Tv, R_sel, C_sel)
                all_lbl.append(lbl)        # (Tv,)
                all_t.append(t_ds)         # (Tv,)

                all_sample_idx.append(sample_idx_ev + base)
                base += t_ds.shape[0]
                used_events += 1


        if used_events == 0:
            raise RuntimeError("No usable events for the requested configuration.")

        signals_tc_rc = np.concatenate(all_sig, axis=0).astype(np.float32)   # (T_total,R_sel,C_sel)
        labels_scalar = np.concatenate(all_lbl, axis=0).astype(np.float32)   # (T_total,)
        times_ms      = np.concatenate(all_t,  axis=0).astype(np.float32)    # (T_total,)
        sample_idx    = np.concatenate(all_sample_idx, axis=0).astype(np.int64)

        # # build sample indices: last index of each valid window
        # sample_idx = _window_indices(np.ones(times_ms.shape[0], dtype=bool), W, hop=hop)
        # if sample_idx.size == 0:
        #     raise RuntimeError("No valid sample windows after filtering.")
        
        if sample_idx.size == 0:
            raise RuntimeError("No valid sample windows after filtering.")

        # stats & standardize
        if self.signal_mean is None or self.signal_stdev is None:
            assert dataset_stage == 'train' or not getattr(self, "train_events", True), f"Dataset_stage: {dataset_stage}"
            m, s = _get_stats_over_windows(signals_tc_rc, sample_idx, W)
            self.signal_mean, self.signal_stdev = float(m), float(s)
            self.save_hyperparameters({'signal_mean': float(m), 'signal_stdev': float(s)})

        if dataset_stage in ['train', 'validation', 'test'] and self.standardize_signals:
            m = np.float32(self.signal_mean)
            s = np.float32(self.signal_stdev)
            eps = np.float32(1e-12)
            print(f"  standardizing: mean={float(m):.4f}, std={float(s):.4f}")
            signals_tc_rc = signals_tc_rc.astype(np.float32, copy=False)
            signals_tc_rc = (signals_tc_rc - m) / np.maximum(s, eps)
            signals_tc_rc = signals_tc_rc.astype(np.float32, copy=False)  # belt-and-suspenders    
                
        # optional row-flip augmentation 
        # if self.do_flip_augmentation:
        #     flipped = signals_tc_rc[:, ::-1, :]      # flip rows
        #     signals_tc_rc = np.concatenate([signals_tc_rc, flipped], axis=0)
        #     # IMPORTANT: v_theta changes sign under poloidal reflection
        #     labels_scalar = np.concatenate([labels_scalar, -labels_scalar], axis=0)
        #     times_ms      = np.concatenate([times_ms,      times_ms],      axis=0)
        #     # NOTE: sample_idx doubles implicitly if you rebuild; simplest is to rebuild:
        #     sample_idx = _window_indices(np.ones(times_ms.shape[0], dtype=bool), W, hop=hop)

        if self.do_flip_augmentation:
            T0 = signals_tc_rc.shape[0]
            flipped = signals_tc_rc[:, ::-1, :]
            signals_tc_rc = np.concatenate([signals_tc_rc, flipped], axis=0)
            labels_scalar = np.concatenate([labels_scalar, -labels_scalar], axis=0)
            times_ms      = np.concatenate([times_ms,      times_ms],      axis=0)
            sample_idx    = np.concatenate([sample_idx,    sample_idx + T0], axis=0)



        # pack dataset / store
        ds = TrainValTest_Dataset_4(
            signals_trc=signals_tc_rc,
            labels_scalar=labels_scalar,
            sample_indices=sample_idx,
            signal_window_size=W,
            times_ms=times_ms,
            n_rows_sel=R_sel,
            n_cols_sel=C_sel,
        )
        if dataset_stage == 'train':
            return ds
        else:
            self.datasets[dataset_stage] = ds
            return None


    def _load_and_preprocess_predict_data_5(self, shot_start_time_indices):
        """
        Prepares the dataset for prediction using the new data structure.
        It loads each prediction event, retrieves the signals and interpolated labels
        from the "interpolated" subgroup for each radial channel, and then generates
        sliding window samples.
        Each sample is a tuple: (signal_window, label, time_point, shot_id, radial_position).
        """
        print("Preparing data for prediction with radial positions")
        data = []  # List to store prediction samples.
        unique_shots = set(shot for shot, _ in shot_start_time_indices)
        print(f"Shots being processed: {unique_shots}")

        with h5py.File(self.data_file, 'r') as h5_file:
            # Filter out shot/event pairs that have both 'signals' and 'times'.
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

                # Load the event-level data.
                event_key = f"{shot}/{start_time}"
                event_data = h5_file[event_key]
                signals = np.array(event_data["signals"], dtype=np.float32)  # shape: (64, time)
                times = np.array(event_data["times"], dtype=np.float32)      # shape: (time,)
                if signals.shape[1] < self.signal_window_size:
                    print(f"Skipping event {event_key}: signal length < window size ({self.signal_window_size})")
                    continue

                # Load shot-level attributes and verify that the "interpolated" subgroup exists.
                shot_group = h5_file[shot]
                if "interpolated" not in shot_group:
                    print(f"Skipping shot {shot}: no interpolated subgroup exists.")
                    continue

                inboard_order = shot_group.attrs.get("inboard_column_channel_order", None)
                if inboard_order is None or len(inboard_order) == 0:
                    print(f"Skipping event {event_key}: Missing or empty inboard_column_channel_order.")
                    continue

                # Reshape the raw signals to shape (time, n_rows, n_cols).
                signals = self.reshape_signals(signals, inboard_order)

                # Apply the optional bandpass filter.
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter")
                    signals = self.apply_bandpass_filter(signals)

                # Retrieve radial positions.
                r_position = shot_group.attrs.get("r_position", None)
                if r_position is not None and len(r_position) == 64:
                    radial_positions = r_position.reshape(8, 8)[0, :].astype(np.float32)
                else:
                    print(f"Warning: Radial positions missing or malformed for shot {shot}.")
                    radial_positions = np.zeros(self.n_cols, dtype=np.float32)

                # Optional: Downsample signals and times.
                if getattr(self, "downsample_factor", 1) > 1:
                    if i % 100 == 0:
                        print(f"  Downsampling signals by factor {self.downsample_factor}")
                    signals = signals[::self.downsample_factor, :, :]
                    times = times[::self.downsample_factor]

                total_time = signals.shape[0]
                # For each radial channel, process prediction samples.
                for col in range(self.n_cols):
                    col_signals = signals[:, :, col]  # shape: (total_time, n_rows)

                    # Load the interpolated data for the current radial channel.
                    if "interpolated" in shot_group and f"radial_{col}" in shot_group["interpolated"]:
                        interp_grp = shot_group["interpolated/radial_" + str(col)]
                        col_label_times = np.array(interp_grp["label_times"], dtype=np.float32)      # Low-res, e.g. shape (N_interp,)
                        col_target = np.array(interp_grp["vZ"], dtype=np.float32)                    # Low-res 1D array.
                        # For prediction, you can choose to use a helper specific for predict,
                        # or reuse the same one as training. For this example, we use a helper:
                        col_labels = self._filter_and_transform_predict_labels(col_label_times, col_target, times)
                    else:
                        print(f"Skipping column {col} in shot {shot}: interpolated data missing for this channel.")
                        continue

                    # Skip if the aligned label array is entirely NaN.
                    if np.isnan(col_labels).all():
                        continue

                    # Remove NaN labels: create a valid mask (the helper should produce an array of length equal to event times).
                    valid_mask = ~np.isnan(col_labels)
                    col_signals = col_signals[valid_mask, :]  # shape: (num_valid, n_rows)
                    col_labels = col_labels[valid_mask]
                    times_filtered = times[valid_mask]

                    # Standardize the signals using training statistics.
                    col_signals = (col_signals - self.signal_mean) / self.signal_stdev

                    # Determine the total number of valid event time points after filtering.
                    T_valid = col_signals.shape[0]
                    if T_valid < self.signal_window_size:
                        continue

                    # Set prediction window stride. For overlapping windows set stride=1;
                    # for non-overlapping, set stride = signal_window_size.
                    stride = getattr(self, "predict_window_stride", 1)

                    # Generate sliding window samples for this radial channel.
                    for idx_t0 in range(self.signal_window_size - 1, T_valid, stride):
                        start_idx = idx_t0 - self.signal_window_size + 1
                        end_idx = idx_t0 + 1
                        signal_window = col_signals[start_idx:end_idx, :]  # shape: (signal_window_size, n_rows)
                        label = col_labels[idx_t0]
                        time_point = times_filtered[idx_t0]
                        data.append((signal_window, label, time_point, shot, radial_positions[col]))

            print(f"Prepared {len(data)} samples for prediction.")
            return PredictDataset_4(data)
        
    def _load_and_preprocess_predict_data_6(self, shot_start_time_indices):
        """
        Prepare prediction samples as sliding windows from the selected (R_sel x C_sel) block.
        Each item: (signal_window[1,W,R_sel,C_sel], label_scalar, time_ms, shot_id, event_id)
        - label_scalar is vθ(ψ_target) aligned to the window end time (or NaN if unavailable).
        - No augmentations applied here.
        """
        import numpy as np, h5py, torch
        from scipy.signal import decimate, resample_poly
        from fractions import Fraction
        import time

        t_start = time.time()

        # -------- helpers (scoped) --------
        def _format_psi_key(psi: float) -> str:
            s = f"{psi:.3f}".rstrip('0').rstrip('.')
            return "psi_" + s.replace('.', 'p')

        def _guess_times_ms(times: np.ndarray) -> np.ndarray:
            if times.size < 2: return times.astype(np.float32)
            dt = np.median(np.diff(times))
            # 1 MHz sampling is often stored in microseconds (dt≈1); convert to ms.
            if 0.5 <= dt <= 2.0 and times.max() > 5e3:
                return (times / 1000.0).astype(np.float32)
            return times.astype(np.float32)

        def _downsample_1d(arr, orig_fs, target_fs, axis=0):
            if orig_fs == target_fs: return arr
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                return decimate(arr, q, ftype='fir', axis=axis, zero_phase=True)
            frac = Fraction(target_fs / orig_fs).limit_denominator(64)
            return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

        def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
            x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
            new_len = x.shape[0]
            t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
            return x.astype(np.float32), t

        def _align_label_scalar_at_times(t_ms, label_t_ms, label_vals, tol_ms=None):
            # nearest neighbor, optional tolerance
            t = np.asarray(t_ms, float)
            lt = np.asarray(label_t_ms, float)
            lv = np.asarray(label_vals, float)
            if t.size == 0 or lt.size == 0:
                return np.full(t.shape, np.nan, dtype=np.float32)

            idx  = np.searchsorted(lt, t, side='left')
            idx0 = np.clip(idx - 1, 0, lt.size - 1)
            idx1 = np.clip(idx,       0, lt.size - 1)
            d0   = np.abs(t - lt[idx0])
            d1   = np.abs(t - lt[idx1])
            use0 = d0 <= d1
            nn   = np.where(use0, idx0, idx1)
            out  = lv[nn].astype(np.float32)

            if tol_ms is not None:
                dnn = np.where(use0, d0, d1)
                out[dnn > tol_ms] = np.nan
            return out

        def _gather_shot_windows_mask(shot: str, t_ms: np.ndarray):
            # Support either: shot_time_windows = {shot: [(t0,t1), ...]}
            # or legacy shot_radial_time_windows = {shot: {col: [(t0,t1), ...], ...}}
            if getattr(self, "shot_time_windows", None) and shot in self.shot_time_windows:
                mask = np.zeros_like(t_ms, dtype=bool)
                for t0, t1 in self.shot_time_windows[shot]:
                    mask |= (t_ms >= t0) & (t_ms <= t1)
                return mask
            if getattr(self, "shot_radial_time_windows", None) and shot in self.shot_radial_time_windows:
                mask = np.zeros_like(t_ms, dtype=bool)
                for _col, wins in self.shot_radial_time_windows[shot].items():
                    for t0, t1 in wins:
                        mask |= (t_ms >= t0) & (t_ms <= t1)
                return mask
            return np.ones_like(t_ms, dtype=bool)

        # ---- canonical selections (on 8x8 after padding) ----
        W      = int(self.signal_window_size)
        hop    = int(getattr(self, "predict_window_stride", 1))
        psi_key = _format_psi_key(self.label_target_psi)

        row_idx = np.arange(8)[self.row_offset::self.row_stride]           # rows on canonical grid
        col_idx = self._cols_from_spec(self.block_cols, n=8)               # cols on canonical grid
        R_sel, C_sel = row_idx.size, col_idx.size

        orig_fs = float(self.sampling_frequency_hz)  # ~1e6
        tgt_fs  = float(self.target_sampling_hz)

        # Storage (lists of numpy arrays / scalars)
        win_list, lbl_list, t0_list, shot_list, event_list = [], [], [], [], []

        with h5py.File(self.data_file, 'r') as h5:
            if len(shot_start_time_indices) >= 5:
                print("  [predict] head events:", shot_start_time_indices[:5])

            # Validate events exist and have signals/times
            valid_events = []
            for shot, event in shot_start_time_indices:
                skey = f"{shot}/{event}"
                if skey in h5 and 'signals' in h5[skey] and 'times' in h5[skey]:
                    valid_events.append((shot, event))
                else:
                    print(f"  [predict] skipping {skey}: missing 'signals'/'times' or event group")

            for i, (shot, event) in enumerate(valid_events):
                if i % 100 == 0:
                    print(f"  [predict] {i:04d}/{len(valid_events):04d}  shot={shot}  event={event}")

                skey = f"{shot}/{event}"
                grp  = h5[skey]
                shot_grp = h5[str(shot)]

                # --- load raw ---
                sig_xt = np.array(grp['signals'], dtype=np.float32)  # (X, T)
                times  = np.array(grp['times'],   dtype=np.float64)  # (T,)
                if sig_xt.shape[1] < W:
                    continue

                # --- robust reshape -> (T, R_full, C_full) ---
                order_attr = shot_grp.attrs.get('inboard_column_channel_order', None)
                sig_trc = self.reshape_signals_2(sig_xt, order_attr)          # (T, R_full, C_full)
                t_ms    = _guess_times_ms(times)

                # --- time cropping before downsample ---
                time_mask = np.ones_like(t_ms, dtype=bool)
                if self.start_time_ms is not None: time_mask &= (t_ms >= self.start_time_ms)
                if self.end_time_ms   is not None: time_mask &= (t_ms <= self.end_time_ms)
                sig_trc = sig_trc[time_mask];  t_ms = t_ms[time_mask]
                if sig_trc.shape[0] < W:
                    continue

                # --- pad to 8x8, then take your sub-grid ---
                sig_trc = self._pad_to_full_grid(sig_trc, target_R=8, target_C=8)  # (T,8,8)
                sig_trc = sig_trc[:, row_idx[:, None], col_idx[None, :]]           # (T,R_sel,C_sel)

                # --- optional bandpass on (T,R,C) ---
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    sig_trc = self.apply_bandpass_filter(sig_trc)

                # --- downsample to target rate ---
                sig_ds, t_ds = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)    # sig: (T_ds,R_sel,C_sel)

                # --- per-shot window mask (if you have one; otherwise all True) ---
                if hasattr(self, "_gather_shot_windows_mask"):
                    shot_mask = _gather_shot_windows_mask(str(shot), t_ds)
                else:
                    shot_mask = np.ones_like(t_ds, dtype=bool)

                # --- labels at fixed ψ (optional in predict; set NaN if missing) ---
                label_vec = None
                # --- labels at fixed ψ (optional in predict) ---
                if "interpolated_psi" in shot_grp and psi_key in shot_grp["interpolated_psi"]:
                    psi_grp = shot_grp["interpolated_psi"][psi_key]
                    lt  = np.array(psi_grp["label_times"], dtype=np.float32)
                    lv  = np.array(psi_grp["vZ"],          dtype=np.float32)
                    label_vec = _align_label_scalar_at_times(t_ds, lt, lv, tol_ms=None)  # (T_ds,)
                else:
                    # no labels for this shot → fill with NaNs so indexing is always valid
                    label_vec = np.full(t_ds.shape, np.nan, dtype=np.float32)

                # ---- build valid indices per-event (avoid crossing event boundaries) ----
                valid = shot_mask
                if label_vec is not None:
                    # no need to drop NaNs for predict; we can carry NaNs to evaluate later
                    pass

                # slide within this event only
                # t0 index (inclusive) is the window end
                t0_candidates = np.arange(W-1, sig_ds.shape[0], hop, dtype=int)
                t0_candidates = t0_candidates[ valid[t0_candidates] ]  # respect mask

                if t0_candidates.size == 0:
                    continue

                # --- standardize now (use train stats) ---
                if self.standardize_signals:
                    assert self.signal_mean is not None and self.signal_stdev is not None, \
                        "Predict requires training signal_mean/signal_stdev to standardize inputs."
                    sig_ds = (sig_ds - self.signal_mean) / max(self.signal_stdev, 1e-12)

                # --- emit windows ---
                for t0 in t0_candidates:
                    win = sig_ds[t0-W+1:t0+1, :, :]       # (W,R_sel,C_sel)
                    lbl = label_vec[t0]   # already float32 (may be NaN)
                    t0_ms = np.float32(t_ds[t0])

                    win_list.append(win[None, ...])       # (1,W,R_sel,C_sel) add channel dim here
                    lbl_list.append(lbl)
                    t0_list.append(t0_ms)
                    shot_list.append(shot)
                    event_list.append(event)

        # Pack numpy arrays
        if len(win_list) == 0:
            print("[predict] no samples prepared.")
        windows   = np.asarray(win_list, dtype=np.float32)      # (N,1,W,R_sel,C_sel)
        labels    = np.asarray(lbl_list, dtype=np.float32)      # (N,)
        times_ms  = np.asarray(t0_list, dtype=np.float32)       # (N,)
        shots     = np.asarray(shot_list)
        events    = np.asarray(event_list)

        print(f"[predict] prepared {windows.shape[0]} windows: shape per-window = {windows.shape[1:]}")
        return PredictDataset_5(
            windows=windows,
            labels=labels,
            times_ms=times_ms,
            shots=shots,
            events=events,
        )

    def _format_psi_key(psi: float) -> str:
        s = f"{psi:.3f}".rstrip('0').rstrip('.')
        return "psi_" + s.replace('.', 'p')

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

    def _filter_and_transform_labels_5(self, label_times, target, uncertainty, times):
        """
        Given:
        - label_times: 1D array of the interpolated (low resolution) label times (in ms)
        - target: 1D array of the corresponding vZ values (interpolated)
        - uncertainty: 1D array of uncertainties for the interpolated data
        - times: 1D array of the event-level times (e.g. in µs; if in µs, consider converting to ms)

        This function returns an aligned 1D array of labels with the same length as 'times',
        where each event time is mapped to the nearest label time. It also applies an uncertainty filter,
        clipping, and (if needed) standardization or normalization.
        """
        # If event times are in µs but label_times are in ms, convert event times:
        # times = times / 1000.0

        # Use searchsorted to map each event time to an index in the label_times array.
        indices = np.searchsorted(label_times, times, side='left')
        # Ensure indices are within bounds.
        indices[indices >= len(label_times)] = len(label_times) - 1
        # For each time, check if the previous label is closer.
        diff_current = np.abs(times - label_times[indices])
        diff_prev = np.abs(times - label_times[np.clip(indices - 1, 0, len(label_times) - 1)])
        use_prev = (indices > 0) & (diff_prev < diff_current)
        indices[use_prev] -= 1

        # Apply uncertainty filtering on the low-resolution target.
        valid_unc = uncertainty <= self.vZ_uncertainty_threshold
        # Create a copy of the target and set values to NaN if not valid.
        target_filtered = target.copy()
        target_filtered[~valid_unc] = np.nan

        # Now produce an aligned label array for each event time.
        labels_aligned = target_filtered[indices]

        # Step 3: Clip labels if desired.
        if self.clip_labels:
            labels_aligned = np.clip(labels_aligned, self.labels_lower_bound, self.labels_upper_bound)

        # Step 4: Standardize or normalize labels if desired.
        if self.standardize_labels:
            labels_aligned = (labels_aligned - self.label_mean) / self.label_std
        elif self.normalize_labels:
            labels_aligned = -1 + 2 * (labels_aligned - self.label_min) / (self.label_max - self.label_min)

        return labels_aligned.astype(np.float32)

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
        # labels_aligned = target[label_indices, :self.n_cols]  # shape (time, n_cols)
        labels_aligned = target[label_indices]  # shape (time,)

        return labels_aligned.astype(np.float32)

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

    def _pad_to_full_grid(self, sig_trc: np.ndarray, target_R: int = 8, target_C: int = 8) -> np.ndarray:
        """
        Pad (T, R, C) to (T, target_R, target_C) with zeros on the row/col tails if needed.
        """
        import numpy as np
        T, R, C = sig_trc.shape
        if R == target_R and C == target_C:
            return sig_trc
        out = np.zeros((T, target_R, target_C), dtype=sig_trc.dtype)
        r_copy = min(R, target_R)
        c_copy = min(C, target_C)
        out[:, :r_copy, :c_copy] = sig_trc[:, :r_copy, :c_copy]
        return out

    def reshape_signals_2(self, sig_xt: np.ndarray, inboard_column_channel_order) -> np.ndarray:
        """
        Map raw BES channels (X,T) -> (T, R, C) using the provided inboard column channel order.

        order = list of length R in which each entry is the channel id for the inboard (radially inner) column
                at that poloidal row. Can be 1-based or 0-based, rows may be out of order. Columns are then
                contiguous outward: idx(row r, col c) = base[r] + c.
        """
        import numpy as np

        sig_xt = np.asarray(sig_xt, dtype=np.float32)  # (X, T)
        if sig_xt.ndim != 2:
            raise ValueError(f"sig_xt must be 2D (X,T), got {sig_xt.shape}")
        X, T = sig_xt.shape

        if inboard_column_channel_order is None:
            raise ValueError("inboard_column_channel_order is required.")

        base = np.asarray(inboard_column_channel_order).ravel().astype(int)  # length R (7 or 8)
        if base.size not in (7, 8):
            raise ValueError(f"inboard_column_channel_order length must be 7 or 8, got {base.size}")

        # Auto-detect 1-based vs 0-based and normalize to 0-based
        if base.min() >= 1 and base.max() <= X:
            base0 = base - 1
        elif base.min() >= 0 and base.max() < X:
            base0 = base
        else:
            raise ValueError(f"order values out of range for X={X}: [{base.min()}, {base.max()}]")

        R = base0.size
        # Maximum universally valid contiguous columns (to avoid OOB if a base is near the end)
        max_cols_possible = int(np.min(X - base0))
        if max_cols_possible <= 0:
            raise ValueError(f"BES mapping invalid: min(X - base0)={max_cols_possible}")

        C = min(8, max_cols_possible)  # nominal is 8

        # Build indices: idx[r, c] = base0[r] + c
        offsets = np.arange(C, dtype=int)[None, :]
        idx_rc = base0[:, None] + offsets  # (R, C)

        # Gather and reshape to (T, R, C)
        sig_selected = sig_xt[idx_rc.reshape(-1), :]   # (R*C, T)
        sig_rct = sig_selected.reshape(R, C, T)        # (R, C, T)
        sig_trc = np.transpose(sig_rct, (2, 0, 1))     # (T, R, C)
        return sig_trc.astype(np.float32)

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
    
    def radial_key(self, r):
        # Multiply by 1e5 and round to convert the float to an integer key.
        return int(round(r))

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