import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch

from bes_data.confinement_data_tools.make_labels import make_labels
from bes_ml.base.data import MultiSourceDataset


class ConfinementDataset(MultiSourceDataset):
    def __init__(self,
                 signal_window_size=128,
                 batch_size=64,
                 logger=None,
                 **kwargs
                 ):
        """PyTorch dataset class to get the ELM data and corresponding labels
        according to the sample_indices. The signals are grouped by `signal_window_size`
        which stacks the time data points and return a data chunk of size:
        (`signal_window_sizex8x8`). The dataset also returns the label which
        corresponds to the label of the last time step of the chunk. Implements weak shuffling,
        i.e. each batch is sampled randomly, however, the data points within a batch are contiguous.

        :param signal_window_size: Signal window size.
        :param logger: Logger object to log the dataset creation process.
        :type logger: logging.getLogger
        """

        if logger is None:
            logger = logging.getLogger(__file__)

        data_location = Path(kwargs.pop('data_location'))
        if data_location.stem != 'labeled_datasets':
            data_location = data_location / 'labeled_datasets'

        super().__init__(signal_window_size=signal_window_size,
                         batch_size=batch_size,
                         logger=logger,
                         data_location=data_location,
                         **kwargs)

        if not self.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.batch_size + self.signal_window_size))
            self.hf2np_labels = np.empty((self.batch_size + self.signal_window_size,))
            self.hf2np_time = np.empty((self.batch_size + self.signal_window_size,))

    def __getitem__(self, index: list):
        if self.dataset_to_ram:
            return self._get_from_ram(index)
        else:
            return self._get_from_hdf5(index)

    def _get_from_ram(self, index):

        hf = self.signals[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_labels = self.labels[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_time = self.time[np.nonzero(self.valid_indices <= index[0])[0][-1]]

        # Adjust index relative to specific HDF5
        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1].astype(int)
        hf_index = [i - idx_offset + self.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.signal_window_size, hf_index[0])) + hf_index

        signal_windows = self._roll_window(hf[hf_index], self.signal_window_size, self.batch_size)
        labels = hf_labels[hf_index[-self.batch_size:]]

        batch_labels = hf_labels[hf_index]
        batch_time = hf_time[hf_index]
        if not all(batch_labels == labels[0]) or not all(np.diff(batch_time) <= 0.2):
            #adjust index for bad labels
            if not all(batch_labels == labels[0]):
                i_offset = np.argmax(batch_labels != batch_labels[0]) + 1
            else:
                i_offset = np.argmax(np.diff(batch_time) >= 2e-3) + 1

            index = [i + i_offset for i in index]
            self._get_from_ram(index)

        return torch.tensor(signal_windows, dtype=torch.float32).unsqueeze(1), torch.tensor(labels, dtype=torch.float32)

    def _get_from_hdf5(self, index):
        try:
            # Get correct index with respect to HDF5 file.
            hf = self.hf_opened[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        except TypeError:
            raise AttributeError('HDF5 files have not been opened! Use TurbulenceDataset.open() ')

        # Adjust index relative to specific HDF5
        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1]
        hf_index = [i - idx_offset + self.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.signal_window_size, hf_index[0])) + hf_index
        hf['signals'].read_direct(self.hf2np_signals, np.s_[:, hf_index], np.s_[...])
        signal_windows = self._roll_window(self.hf2np_signals.transpose(), self.signal_window_size, self.batch_size)

        hf['labels'].read_direct(self.hf2np_labels, np.s_[hf_index], np.s_[...])
        hf['time'].read_direct(self.hf2np_time, np.s_[hf_index], np.s_[...])

        labels = self.hf2np_labels[-self.batch_size:]

        if not all(self.hf2np_labels == labels[0]) or not all(np.diff(self.hf2np_time) <= 0.2):
            # adjust index for bad labels
            i_offset = np.argmax(self.hf2np_labels != self.hf2np_labels[0])
            index = [i + i_offset for i in index]
            self._get_from_hdf5(index)

        return torch.tensor(signal_windows, dtype=torch.float32).unsqueeze(1), torch.tensor(labels)

    def _get_f_lengths(self):
        """
        Get lengths of all hdf5 files.
        :rtype: np.array
        """
        length_arr = []
        if self.istest_ and self.labels is not None:
            # this might be important to change for multi-shot
            for f in self.labels:
                length_arr.append(f)
        elif not self.istest_:
            for f in self.input_files:
                with h5py.File(f, 'r') as ds:
                    length_arr.append(np.array(ds['labels']))
        else:
            return [0]

        fs = []
        for l in length_arr:
            n_labels = len(np.unique(l))
            discontinuous_labels = self.batch_size * n_labels
            window_start_offset = self.signal_window_size + self.batch_size
            fs.append(np.around(len(l) * self.frac_) - window_start_offset - discontinuous_labels)
        return np.array(fs, dtype=int)

    def load_datasets(self,
                      signals=None,
                      labels=None,
                      time=None,
                      ):
        """Load datasets into RAM"""

        # use data loaded into dataset externally
        if signals is not None and labels is not None:
            assert max(signals.shape) == max(labels.shape), f"Signals and Labels must be same length. Got {len(signals)} and {len(labels)}"
            self.signals = [signals]
            self.labels = [labels]
            self.time = [time]
            self.f_lengths = self._get_f_lengths()
            return

        if self.istrain_:
            s = 'Training '
        elif self.isvalid_:
            s = 'Validation '
        else:
            s = ' '
        self.logger.info(f"Loading {s}datasets into RAM.")
        signals, labels, time = [], [], []

        self.open()
        if len(self.shot_nums) < 3:
            for hf in self.hf_opened:
                n_indices = max(hf['signals'].shape)
                # Indices for start and stop of validation and test sets
                i_start = np.floor((1 - (self.fraction_test + self.fraction_validation)) * n_indices).astype(int)
                i_stop = np.floor((1 - self.fraction_test) * n_indices).astype(int)

                if self.isvalid_:
                    sx = np.s_[i_start:i_stop]
                    sx_s = np.s_[:, i_start:i_stop]
                elif self.istest_:
                    sx = np.s_[i_stop:n_indices]
                    sx_s = np.s_[:, i_stop:n_indices]
                elif self.istrain_:
                    sx = np.s_[0:i_start]
                    sx_s = np.s_[:, 0:i_start]

                # read_direct is faster and more memory efficient
                arr_len = sx.stop - sx.start
                hf2np_s = np.empty((64, arr_len))
                hf2np_l = np.empty((arr_len,))
                hf2np_t = np.empty((arr_len,))

                hf['signals'].read_direct(hf2np_s, sx_s, np.s_[...])
                hf['labels'].read_direct(hf2np_l, sx, np.s_[...])
                hf['time'].read_direct(hf2np_t, sx, np.s_[...])
                signals.append(hf2np_s.transpose())
                labels.append(hf2np_l)
                time.append(hf2np_t)
        else:
            for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
                print(f'\rProcessing shot {sn} ({i+1}/{len(self.shot_nums)})', end=' ')
                signals_np = np.array(hf['signals']).transpose()
                labels_np = np.array(hf['labels'])
                time_np = np.array(hf['time'])
                print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
                signals.append(signals_np)
                labels.append(labels_np)
                time.append(time_np)
        self.close()

        self.signals = signals
        self.labels = labels
        self.time = time
        self.logger.info(f'{s}datasets loaded successfully.')

        return