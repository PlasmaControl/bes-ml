from pathlib import Path
import dataclasses
import re
import traceback

import h5py
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, BatchSampler

from bes_data.sample_data import sample_data_dir
try:
    from ..base.train_base import Trainer_Base_Dataclass
    from ..base.models import Multi_Features_Model_Dataclass
    from ..base.sampler import RandomBatchSampler
except ImportError:
    from bes_ml.base.train_base import Trainer_Base_Dataclass
    from bes_ml.base.models import Multi_Features_Model_Dataclass
    from bes_ml.base.sampler import RandomBatchSampler


@dataclasses.dataclass(eq=False)
class Confinement_Data_v2(
    Trainer_Base_Dataclass,
    Multi_Features_Model_Dataclass,
):
    data_location: Path|str = sample_data_dir / 'kgill_data' #location of stored data
    dataset_to_ram: bool = True # Load datasets to ram
    batch_size: int = 64  # power of 2, like 16-128
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    num_workers: int = None  # number of subprocess workers for pytorch dataloader
    pin_memory: bool = True  # data loader pinned memory
    seed: int = None  # RNG seed for deterministic, reproducible shuffling (ELMs, sample indices, etc.)
    label_type: np.int8 | np.float32 = dataclasses.field(default=None, init=False)

    def _prepare_data(self) -> None:
        self.data_location = Path(self.data_location)
        if self.data_location.stem != 'labeled_datasets':
            self.data_location = self.data_location / 'labeled_datasets'
        self.data_location = Path(self.data_location).resolve()
        assert self.data_location.exists(), f"{self.data_location} does not exist"

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

        self.logger.info(f'Loading files from {self.data_location}')
        self.shot_nums, self.input_files = self._retrieve_filepaths()
        self.logger.info(f'Found {len(self.input_files)} files!')
        self._make_datasets()
        self._make_data_loaders()

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
        input_files = [shots[key] for key in sorted(shots.keys())]
        shot_nums = list(sorted(shots.keys()))
        return shot_nums, input_files
    
    def _make_datasets(self):
        """
        Splits full dataset into train and test sets. Returns copies of self.
        :param test_frac: Fraction of dataset for test set.
        :param seed: Numpy random seed. Default None.
        :return: train_set, test_set
        :rtype: tuple(TurbulenceDataset, TurbulenceDataset)
        """
        np.random.seed(self.seed)
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

        test_files = shots_files[test_idx]
        valid_files = shots_files[valid_idx]
        train_files = shots_files[train_idx]

        # train = copy.deepcopy(self)
        train_dataset = ConfinementDataset(
            data_location=self.data_location,
            signal_window_size=self.signal_window_size,
            batch_size=self.batch_size,
            fraction_test=self.fraction_test,
            fraction_validation=self.fraction_validation,
            dataset_to_ram=self.dataset_to_ram,
            state='train',
        )
        train_dataset.shot_nums = [i[0] for i in train_files]
        train_dataset.input_files = [i[1] for i in train_files]
        train_dataset.f_lengths = train_dataset._get_f_lengths()
        train_dataset.valid_indices = np.cumsum(np.concatenate((np.array([0]), train_dataset.f_lengths)))[:-1]

        # valid = copy.deepcopy(self)
        valid_dataset = ConfinementDataset(
            data_location=self.data_location,
            signal_window_size=self.signal_window_size,
            batch_size=self.batch_size,
            fraction_test=self.fraction_test,
            fraction_validation=self.fraction_validation,
            dataset_to_ram=self.dataset_to_ram,
            state='valid',
        )
        valid_dataset.shot_nums = [i[0] for i in valid_files]
        valid_dataset.input_files = [i[1] for i in valid_files]
        valid_dataset.f_lengths = valid_dataset._get_f_lengths()
        valid_dataset.valid_indices = np.cumsum(np.concatenate((np.array([0]), valid_dataset.f_lengths)))[:-1]

        # test = copy.deepcopy(self)
        test_dataset = ConfinementDataset(
            data_location=self.data_location,
            signal_window_size=self.signal_window_size,
            batch_size=self.batch_size,
            fraction_test=self.fraction_test,
            fraction_validation=self.fraction_validation,
            dataset_to_ram=self.dataset_to_ram,
            state='test',
        )
        test_dataset.shot_nums = [i[0] for i in test_files]
        test_dataset.input_files = [i[1] for i in test_files]
        test_dataset.f_lengths = valid_dataset._get_f_lengths()
        test_dataset.valid_indices = np.cumsum(np.concatenate((np.array([0]), test_dataset.f_lengths)))[:-1]

        assert all([len(ds) > 0 for ds in [train_dataset, test_dataset, valid_dataset]]), 'There is not enough data to split datasets.'

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        if self.dataset_to_ram:
            # Load datasets into ram
            self.train_dataset.load_datasets()
            self.valid_dataset.load_datasets()

    def _make_data_loaders(self):
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(
                RandomBatchSampler(
                    self.train_dataset,
                    self.batch_size,
                    self.signal_window_size,
                ),
                batch_size=self.batch_size,
                drop_last=True,
            )
        )
        self.validation_data_loader = DataLoader(
            self.valid_dataset,
            batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(
                RandomBatchSampler(
                    self.valid_dataset,
                    self.batch_size,
                    self.signal_window_size,
                ),
                batch_size=self.batch_size,
                drop_last=True,
            )
        )


class ConfinementDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        signal_window_size=128,
        batch_size=64,
        data_location=None,
        fraction_test=0.15,
        fraction_validation=0.1,
        dataset_to_ram=True,
        state: str = None,
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

        self.data_location = data_location
        self.signal_window_size = signal_window_size
        self.batch_size = batch_size
        self.fraction_test = fraction_test
        self.fraction_validation = fraction_validation

        assert Path(self.data_location).exists(), f'{self.data_location} does not exist'

        self.shot_nums, self.input_files = self._retrieve_filepaths()
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

        if not self.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.batch_size + self.signal_window_size))
            self.hf2np_labels = np.empty((self.batch_size + self.signal_window_size,))
            self.hf2np_time = np.empty((self.batch_size + self.signal_window_size,))

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

    def _roll_window(self, arr, sws, bs) -> np.ndarray:
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

    def __len__(self):
        return int(sum(self.f_lengths))

    def __getitem__(self, index: list):
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
            self.close()
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, tb)
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
            for f in self.hf_opened:
                f.close()
        self.hf_opened = None
        return

    def _get_from_ram(self, index):

        hf = self.signals[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_labels = self.labels[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_time = self.time[np.nonzero(self.valid_indices <= index[0])[0][-1]]

        # Adjust index relative to specific HDF5
        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1].astype(int)
        hf_index = [i - idx_offset + self.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.signal_window_size + 1, hf_index[0])) + hf_index

        signal_windows = self._roll_window(hf[hf_index], self.signal_window_size, self.batch_size)
        labels = hf_labels[hf_index[-self.batch_size:]]

        assert signal_windows.shape[0] == self.batch_size and signal_windows.shape[1] == self.signal_window_size
        assert labels.shape[0] == self.batch_size

        batch_labels = hf_labels[hf_index]
        batch_time = hf_time[hf_index]
        print("!!!!!!!!!!!!!!!!!!!!!!",batch_labels, batch_time, labels)
        # if not all(batch_labels == labels[0]) or not all(np.diff(batch_time) <= 0.2):
        #     #adjust index for bad labels
        #     if not all(batch_labels == labels[0]):
        #         i_offset = np.argmax(batch_labels != batch_labels[0]) + 1
        #     else:
        #         i_offset = np.argmax(np.diff(batch_time) >= 2e-3) + 1

        #     index = [i + i_offset for i in index]
        #     self._get_from_ram(index)

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
                hf2np_l = np.empty((arr_len,4))
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

        return
