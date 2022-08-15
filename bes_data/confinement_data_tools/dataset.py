import argparse
import logging
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch

from bes_ml.base.data import MultiSourceDataset


class TurbulenceDataset(MultiSourceDataset):
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

        super().__init__(signal_window_size=signal_window_size,
                         batch_size=batch_size,
                         logger=logger,
                         **kwargs)

        if not self.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.batch_size + self.signal_window_size - 1))
            self.hf2np_labels = np.empty((self.batch_size,))


    def _get_from_hdf5(self, index):
        try:
            # Get correct index with respect to HDF5 file.
            hf = self.hf_opened[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        except TypeError:
            raise AttributeError('HDF5 files have not been opened! Use TurbulenceDataset.open() ')

        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1]# Adjust index relative to specific HDF5
        hf_index = [i - idx_offset + self.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.signal_window_size + 1, hf_index[0])) + hf_index
        hf['signals'].read_direct(self.hf2np_signals, np.s_[:, hf_index], np.s_[...])
        signal_windows = self._roll_window(self.hf2np_signals.transpose(), self.signal_window_size, self.batch_size)

        hf['labels'].read_direct(self.hf2np_labels, np.s_[hf_index[-self.batch_size:]], np.s_[...])

        return torch.tensor(signal_windows).unsqueeze(1), torch.tensor(self.hf2np_labels)


    def _retrieve_filepaths(self, input_dir=None):
        """
        Get filenames of all labeled files.
        :param input_dir: (optional) Change the input data directory.
        :return: all shot numbers, all shot file paths.
        :rtype: (list, list)
        """
        if input_dir:
            self.data_location = input_dir
        dir = Path(self.data_location) / 'labeled_datasets'
        if not dir.exists() or not any(Path(dir).iterdir()):
            self._make_labels()
        shots = {}
        for file in (dir.iterdir()):
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

    def _get_f_lengths(self):
        """
        Get lengths of all hdf5 files.
        :rtype: np.array
        """
        fs = []
        for f in self.input_files:
            with h5py.File(f, 'r') as ds:
                fs.append(len(ds['labels']) - self.signal_window_size - self.batch_size)
        return np.array(fs, dtype=int)

    def _make_labels(self,
                     base_dir: Union[str, Path] = None,
                     df_name: str = 'confinement_database.xlsx',
                     data_dir: Union[str, Path] = 'turbulence_data',
                     labeled_dir: Union[str, Path] = 'labeled_datasets'):
        """
        Function to create labeled datasets for turbulence regime classification.
        Shot data is sourced from base_dir / turbulence_data.
        Resulting labeled datasets are stored as HDF5 files in base_dir / data / labeled_datasets.
        :param base_dir: Home directory of project. Should contain 'confinement_database.xlsx' and 'turbulence_data/'
        :param df_name: Name of the confinement regime database file.
        :param data_dir: Path to datasets (rel. to base_dir or specify whole path.)
        :param labeled_dir: Path to labeled datasets (rel. to data_dir or specify whole path.)
        :return: None
        """
        # Find already labeled datasets
        if base_dir:
            # Pathify all directories
            if Path(data_dir).exists():
                data_dir = Path(data_dir)
            else:
                data_dir = Path(base_dir) / data_dir
        else:
            data_dir = self.data_location
            base_dir = data_dir.parent

        if Path(labeled_dir).exists():
            labeled_dir = Path(labeled_dir)
        else:
            labeled_dir = Path(data_dir) / labeled_dir

        labeled_dir.mkdir(exist_ok=True)
        labeled_shots = {}
        for file in (labeled_dir.iterdir()):
            try:
                shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
            except IndexError:
                continue
            if shot_num not in labeled_shots.keys():
                labeled_shots[shot_num] = file

        # Find unlabeled datasets (shots not in base_dir/data/labeled_datasets)
        shots = {}
        for file in (data_dir.iterdir()):
            try:
                shot_num = re.findall(r'_(\d+).hdf5', str(file))[0]
            except IndexError:
                continue
            if shot_num not in labeled_shots.keys():
                shots[shot_num] = file

        if len(shots) == 0:
            self.logger.info('No new labels to make')
            return
        self.logger.info(f'Making labels for shots {[sn for sn in shots.keys()]}')

        # Read labeled df.
        label_df = pd.read_excel(base_dir / df_name).fillna(0)
        for shot_num, file in shots.items():
            shot_df = label_df.loc[label_df['shot'] == float(shot_num)]
            if len(shot_df) == 0:
                print(f'{shot_num} not in confinement database.')
                continue
            else:
                print(f'Processing shot {shot_num}')

            with h5py.File(file, 'a') as shot_data:
                try:
                    labels = np.array(shot_data['labels']).tolist()
                except KeyError:
                    time = np.array(shot_data['time'])
                    signals = np.array(shot_data['signals'])
                    labels = np.zeros_like(time)

                    for i, row in shot_df.iterrows():
                        tstart = row['tstart (ms)']
                        tstop = row['tstop (ms)']
                        label = row[[col for col in row.index if 'mode' in col]].values.argmax() + 1
                        labels[np.nonzero((time > tstart) & (time < tstop))] = label

                    signals = signals[:, np.nonzero(labels)[0]]
                    time = time[np.nonzero(labels)[0]]
                    labels = labels[np.nonzero(labels)[0]] - 1

            sname = f'bes_signals_{shot_num}_labeled.hdf5'
            with h5py.File(labeled_dir / sname, 'w') as sd:
                sd.create_dataset('labels', data=labels)
                sd.create_dataset('signals', data=signals)
                sd.create_dataset('time', data=time)

        return

    def load_datasets(self):
        """Load datasets into RAM"""
        if self.istrain_:
            s = 'Training '
        elif self.isvalid_:
            s = 'Validation '
        else:
            s = ' '
        self.logger.info(f"Loading {s}datasets into RAM.")
        signals, labels = [], []

        self.open()
        for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
            print(f'\rProcessing shot {sn} ({i+1}/{len(self.shot_nums)})', end=' ')
            signals_np = np.array(hf['signals']).transpose()
            labels_np = np.array(hf['labels'])
            print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
            signals.append(signals_np)
            labels.append(labels_np)
        self.close()

        self.signals = signals
        self.labels = labels
        self.logger.info(f'{s}datasets loaded successfully.')

        return