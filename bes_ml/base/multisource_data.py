import pickle
import re
import copy
from pathlib import Path
import logging
import traceback
from typing import Union
import dataclasses

import h5py
import numpy as np
import torch
import torch.utils.data

from bes_data.sample_data import sample_elm_data_file
try:
    from .train_base import Trainer_Base_Dataclass
    from .utilities import merge_pdfs
except ImportError:
    from bes_ml.base.train_base import _Base_Trainer_Dataclass
    from bes_ml.base.utilities import merge_pdfs

@dataclasses.dataclass(eq=False)
class _MultiSource_Data_Base(Trainer_Base_Dataclass):
    batch_size: int = 64  # power of 2, like 16-128
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing

class MultiSourceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_location=None,
                 output_dir='run_dir',
                 signal_window_size=128,
                 batch_size=64,
                 fraction_test=0.15,
                 fraction_validation=0.1,
                 dataset_to_ram=True,
                 state: str = None,
                 logger: logging.Logger = None,
                 ):

        self.data_location = data_location
        self.output_dir = output_dir
        self.signal_window_size = signal_window_size
        self.batch_size = batch_size
        self.fraction_test = fraction_test
        self.fraction_validation = fraction_validation
        self.logger = logger

        assert Path(self.data_location).exists(), f'{self.data_location} does not exist'
        self.logger.info(f'Loading files from {self.data_location}')

        self.shot_nums, self.input_files = self._retrieve_filepaths()
        self.logger.info(f'Found {len(self.input_files)} files!')
        self.dataset_to_ram = dataset_to_ram if len(self.shot_nums) >= 3 else True

        # Some flags for operations and checks
        self.open_ = False
        self.istrain_ = bool('train' in str(state).lower())
        self.istest_ = bool('test' in str(state).lower())
        self.isvalid_ = bool('valid' in str(state).lower())
        self.frac_ = 1
        self.signals = None
        self.labels = None
        self.time = None
        self.hf_opened = None

        self.f_lengths = self._get_f_lengths()
        self.valid_indices = np.cumsum(np.concatenate((np.array([0]), self.f_lengths)))[:-1]

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

    def _retrieve_filepaths(self, input_dir=None):
        """
        Get filenames of all labeled files.
        :param input_dir: (optional) Change the input data directory.
        :return: all shot numbers, all shot file paths.
        :rtype: (list, list)
        """
        if input_dir:
            self.data_location = input_dir
        data_loc = Path(self.data_location)
        assert data_loc.exists(), f'Directory {data_loc} does not exist. Have you made datasets?'
        shots = {}
        for file in (data_loc.iterdir()):
            try:
                shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
            except IndexError:
                continue
            if shot_num not in shots.keys():
                shots[shot_num] = file

        # Keeps them in the same order for __getitem__
        self.input_files = [shots[key] for key in sorted(shots.keys())]
        self.shot_nums = list(sorted(shots.keys()))

        return self.shot_nums, self.input_files

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

    def train_test_split(self, seed: int = 42):
        """
        Splits full dataset into train and test sets. Returns copies of self.
        :param test_frac: Fraction of dataset for test set.
        :param seed: Numpy random seed. Default None.
        :return: train_set, test_set
        :rtype: tuple(TurbulenceDataset, TurbulenceDataset)
        """
        np.random.seed(seed)
        shots_files = np.array(list(zip(self.shot_nums, self.input_files)))
        valid_idx, train_idx, test_idx = [], [], []
        if len(shots_files) >= 3:
            sf_idx = np.arange(len(shots_files), dtype=np.int32)
            n_valid = int(np.floor(len(shots_files) * self.fraction_validation))
            n_valid = n_valid if n_valid else 1
            n_test = int(np.floor(len(shots_files) * self.fraction_test))
            n_test = n_test if n_test else 1
            valid_and_test = np.random.choice(sf_idx, n_valid+n_test, replace=False)
            train_idx = np.array([i for i in sf_idx if i not in valid_and_test], dtype=np.int32)
            valid_idx = np.random.choice(valid_and_test, n_valid, replace=False)
            test_idx = np.array([i for i in valid_and_test if i not in valid_idx], dtype=np.int32)
        else:
            for s in range(len(shots_files)):
                valid_idx.append(s)
                train_idx.append(s)
                test_idx.append(s)

        test_set = shots_files[test_idx]
        valid_set = shots_files[valid_idx]
        train_set = shots_files[train_idx]

        train = copy.deepcopy(self)
        train._set_state('train')
        train.shot_nums = [i[0] for i in train_set]
        train.input_files = [i[1] for i in train_set]
        train.f_lengths = train._get_f_lengths()
        train.valid_indices = np.cumsum(np.concatenate((np.array([0]), train.f_lengths)))[:-1]

        valid = copy.deepcopy(self)
        valid._set_state('valid')
        valid.shot_nums = [i[0] for i in valid_set]
        valid.input_files = [i[1] for i in valid_set]
        valid.f_lengths = valid._get_f_lengths()
        valid.valid_indices = np.cumsum(np.concatenate((np.array([0]), valid.f_lengths)))[:-1]

        test = copy.deepcopy(self)
        test._set_state('test')
        test.shot_nums = [i[0] for i in test_set]
        test.input_files = [i[1] for i in test_set]
        test.f_lengths = valid._get_f_lengths()
        test.valid_indices = np.cumsum(np.concatenate((np.array([0]), test.f_lengths)))[:-1]

        assert all([len(ds) > 0 for ds in [train, test, valid]]), 'There is not enough data to split datasets.'

        return train, valid, test

    def save(self, output_file: Union[Path,str]):

        if self.dataset_to_ram:
            signals = np.concatenate(self.signals)
            labels = np.concatenate(self.labels)
            try:
                time = np.concatenate(self.time)
            except TypeError:
                time = None
        else:
            self.open()
            signals, labels, time = None, None, None
            for hf in self.hf_opened:
                signals = np.array(hf['signals'])
                labels = np.array(hf['labels'])
                try:
                    time = np.array(hf['time'])
                except TypeError:
                    time = None

        data_dict = {'signals': signals, 'labels': labels, 'time': time}
        with open(output_file, 'w+b') as f:
            pickle.dump(data_dict, f)
        self.close()
        return

    def _get_f_lengths(self):
        raise NotImplementedError
