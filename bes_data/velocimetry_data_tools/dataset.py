import logging
import pickle
import re
from pathlib import Path

import h5py
import numpy as np

from bes_ml.base.elm_data import MultiSourceDataset


class VelocimetryDataset(MultiSourceDataset):
    def __init__(self,
                 signal_window_size=128,
                 batch_size=64,
                 logger: logging.Logger = None,
                 **kwargs):
        """PyTorch dataset class to get the ELM data and corresponding velocimetry calculations.
        The signals are grouped by `signal_window_size` which stacks the time data points
        and return a data chunk of size: (`signal_window_sizex8x8`). The dataset also returns the label which
        corresponds to the label of the last time step of the chunk. Implements weak shuffling,
        i.e. each batch is sampled randomly, however, the data points within a batch are contiguous.

        :param data_location: Directory where velocimetry data is stored.
        :param signal_window_size: Signal window size.
        :param batch_size: Batch size.
        :param logger: Logger object to log the dataset creation process.
        """

        if not logger:
            logger = logging.getLogger('__main__')

        super().__init__(signal_window_size=signal_window_size,
                         batch_size=batch_size,
                         logger=logger,
                         **kwargs)

        if not self.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.batch_size + self.signal_window_size - 1))
            self.hf2np_vZ = np.empty((self.batch_size, 8, 8))
            self.hf2np_vR = np.empty((self.batch_size, 8, 8))

    def _get_from_hdf5(self, index):
        raise NotImplementedError('Can not get from hdf5.')

    def _get_f_lengths(self):
        """
        Get lengths of all hdf5 files.
        :rtype: np.array
        """
        length_arr = []
        if self.istest_ and self.signals is not None:
            # this might be important to change for multi-shot
            for f in self.signals:
                length_arr.append(len(f))
        elif not self.istest_:
            for f in self.input_files:
                with h5py.File(f, 'r') as ds:
                    length_arr.append(ds['signals'].shape[-1])
        else:
            return [0]

        fs = []
        for l in length_arr:
            fs.append(np.around(l * self.frac_) - self.signal_window_size - self.batch_size)
        assert all([l >= 0 for l in fs]), "There are not enough data points to make dataset."
        return np.array(fs)

    def load_datasets(self,
                      signals=None,
                      labels=None,
                      ):
        """Load datasets into RAM"""

        # use data loaded into dataset externally
        if signals is not None and labels is not None:
            assert max(signals.shape) == max(labels.shape), f"Signals and Labels must be same length. Got {len(signals)} and {len(labels)}"
            self.signals = [signals.reshape(-1, 64)]
            self.labels = [labels]
            self.f_lengths = self._get_f_lengths()
            return

        # Only used for displaying strings in console.
        if self.istrain_:
            s = 'Training '
        elif self.isvalid_:
            s = 'Validation '
        else:
            s = ''
        self.logger.info(f"Loading {s}datasets into RAM.")
        signals, labels = [], []

        self.open()
        if len(self.shot_nums) < 3:
            for hf in self.hf_opened:
                n_indices = hf['signals'].shape[-1]
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
                hf2np_vR = np.empty((arr_len, 8, 8))
                hf2np_vZ = np.empty((arr_len, 8, 8))

                hf['signals'].read_direct(hf2np_s, sx_s, np.s_[...])
                hf['vR'].read_direct(hf2np_vR, sx, np.s_[...])
                hf['vZ'].read_direct(hf2np_vZ, sx, np.s_[...])
                signals.append(hf2np_s.transpose())
                labels.append(np.concatenate((hf2np_vR.reshape(-1, 64), hf2np_vZ.reshape(-1, 64)), axis=1))
        else:
            for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
                print(f'\rProcessing shot {sn} ({i + 1}/{len(self.shot_nums)})', end=' ')
                signals_np = np.array(hf['signals']).transpose()
                vr_labels_np = np.array(hf['vR']).reshape((-1, 64))
                vz_labels_np = np.array(hf['vZ']).reshape((-1, 64))
                labels_np = np.concatenate((vr_labels_np, vz_labels_np), axis=1)
                print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
                signals.append(signals_np)
                labels.append(labels_np)
        self.close()

        self.signals = signals
        self.labels = labels
        self.logger.info(f'{s}datasets loaded successfully.')

        return
