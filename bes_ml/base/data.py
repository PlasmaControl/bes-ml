import h5py
import numpy as np
import torch
# multisource imports
import copy
from pathlib import Path
import logging
import traceback
import torch.utils.data


class ELM_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        signals: np.ndarray = None, 
        labels: np.ndarray = None, 
        sample_indices: np.ndarray = None, 
        window_start: np.ndarray = None,
        signal_window_size: int = None,
        prediction_horizon: int = None,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        # self.window_start = window_start
        self.signal_window_size = signal_window_size
        self.prediction_horizon = prediction_horizon if prediction_horizon is not None else 0

    def __len__(self):
        return self.sample_indices.size

    def __getitem__(self, idx: int):
        time_idx = self.sample_indices[idx]
        # BES signal window data
        signal_window = self.signals[
            time_idx : time_idx + self.signal_window_size
        ]
        signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        # label for signal window
        label = self.labels[
            time_idx
            + self.signal_window_size
            + self.prediction_horizon
            - 1
        ]
        label = torch.as_tensor(label)

        return signal_window, label


class MultiSourceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_location=None,
                 output_dir='run_dir',
                 signal_window_size=128,
                 batch_size=64,
                 fraction_test=0.15,
                 fraction_validation=0.1,
                 dataset_to_ram=True,
                 logger: logging.Logger = None,
                 ):

        self.data_location = data_location
        self.output_dir = output_dir
        self.signal_window_size = signal_window_size
        self.batch_size = batch_size
        self.fraction_test = fraction_test
        self.fraction_validation = fraction_validation
        self.dataset_to_ram = dataset_to_ram
        self.logger = logger

        assert Path(self.data_location).exists()
        self.logger.info(f'Loading files from {self.data_location}')

        self.shot_nums, self.input_files = self._retrieve_filepaths()
        self.logger.info(f'Found {len(self.input_files)} files!')

        # Some flags for operations and checks
        self.open_ = False
        self.istrain_ = False
        self.istest_ = False
        self.isvalid_ = False
        self.frac_ = 1

        self.f_lengths = self._get_f_lengths()
        self.valid_indices = np.cumsum(np.concatenate((np.array([0]), self.f_lengths)))[:-1]

        self.signals = None
        self.labels = None
        self.hf_opened = None

    def __len__(self):
        return int(sum(self.f_lengths))

    def __getitem__(self, index: int):
        if self.dataset_to_ram:
            return self._get_from_ram(index)
        else:
            return self._get_from_hdf5(index)

    def __enter__(self):
        if not self.dataset_to_ram:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.open_:
            self.logger.info('Closing all open datasets')
            self.close()
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, tb)
                # return False # uncomment to pass exception through
        return

    def open(self):
        """
        Open all the datasets in self.data_dir for access.
        """
        self.open_ = True
        hf_opened = []
        for f in self.input_files:
            hf_opened.append(h5py.File(f, 'r'))
        self.hf_opened = hf_opened

        return self

    def close(self):
        """
        Close all the data sets previously opened.
        :return:
        """
        if self.open_:
            self.open_ = False
            self.logger.info('Closing all open hdf5 files.')
            for f in self.hf_opened:
                f.close()
        self.hf_opened = None
        return

    def _get_from_ram(self, index):

        hf = self.signals[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_labels = self.labels[np.nonzero(self.valid_indices <= index[0])[0][-1]]

        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1].astype(int)  # Adjust index relative to specific HDF5
        hf_index = [i - idx_offset + self.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.signal_window_size + 1, hf_index[0])) + hf_index

        signal_windows = self._roll_window(hf[hf_index], self.signal_window_size, self.batch_size)
        labels = hf_labels[hf_index[-self.batch_size:]]

        return torch.tensor(signal_windows, dtype=torch.float32).unsqueeze(1), torch.tensor(labels, dtype=torch.float32)

    def _roll_window(self, arr, sws, bs):
        """
        Helper function to return rolling window view of array.
        :param arr: Array to be rolled
        :type arr: np.ndarray
        :param sws: Signal window size
        :type sws: int
        :param bs: Batch size
        :type bs: int
        :return: Rolling window view of array
        :rtype: np.ndarray
        """
        return np.lib.stride_tricks.sliding_window_view(arr.view(), sws, axis=0)\
            .swapaxes(-1, 1)\
            .reshape(bs, -1, 8, 8)

    def _set_state(self, state: str):
        self.istrain_ = False
        self.istest_ = False
        self.isvalid_ = False
        if state == 'train':
            self.istrain_ = True
            self.frac_ = 1 - (self.fraction_validation + self.fraction_test)
        elif state == 'valid':
            self.isvalid_ = True
            self.frac_ = self.fraction_validation
        elif state == 'test':
            self.istest_ = True
            self.frac_ = self.fraction_test
        else:
            pass

    def train_test_split(self, test_frac: float, seed=None):
        """
        Splits full dataset into train and test sets. Returns copies of self.
        :param test_frac: Fraction of dataset for test set.
        :param seed: Numpy random seed. Default None.
        :return: train_set, test_set
        :rtype: tuple(TurbulenceDataset, TurbulenceDataset)
        """
        np.random.seed(seed)
        shots_files = np.array(list(zip(self.shot_nums, self.input_files)))
        test_idx, train_idx = [0], [0]
        if len(shots_files) != 1:
            sf_idx = np.arange(len(shots_files), dtype=np.int32)
            n_test = int(np.floor(len(shots_files) * test_frac))
            n_test = n_test if n_test else 1
            test_idx = np.random.choice(sf_idx, n_test, replace=False)
            train_idx = np.array([i for i in sf_idx if i not in test_idx], dtype=np.int32)

        test_set = shots_files[test_idx]
        train_set = shots_files[train_idx]

        train = copy.deepcopy(self)
        train._set_state('train')
        train.shot_nums = [i[0] for i in train_set]
        train.input_files = [i[1] for i in train_set]
        train.f_lengths = train._get_f_lengths()
        train.valid_indices = np.cumsum(np.concatenate((np.array([0]), train.f_lengths)))[:-1]

        test = copy.deepcopy(self)
        test._set_state('valid')
        test.shot_nums = [i[0] for i in test_set]
        test.input_files = [i[1] for i in test_set]
        test.f_lengths = test._get_f_lengths()
        test.valid_indices = np.cumsum(np.concatenate((np.array([0]), test.f_lengths)))[:-1]

        return train, test

    def _retrieve_filepaths(self):
        raise NotImplementedError

    def _get_f_lengths(self):
        raise NotImplementedError

def elm_data_loader(
    dataset: ELM_Dataset = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader
